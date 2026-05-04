#!/usr/bin/env python3
"""
numeric_eval_by_horizon.py

Computes numeric evaluation metrics for UM-DeepOutbreak flu hospitalization forecasts
by forecast horizon, matched against CDC observed ground-truth data.

Truth source:
    cdc_datafiles.csv  (same file used by pred_vs_truth_epiweeks.py)
    Deduplication: latest as_of revision kept per (location, target_end_date).
    Location matching: submission FIPS code -> truth location_name via loc_mapping.

Metrics (per horizon, across all matched forecast-truth pairs):
    n             : number of forecast-truth pairs with observed truth
    mae           : Mean Absolute Error using median (q0.5) forecast
    rmse          : Root Mean Square Error using median forecast
    coverage_90   : Fraction of truth values within 90% PI [q0.05, q0.95]
    mean_width_90 : Mean width of the 90% PI
    wis           : Mean Weighted Interval Score (Bracher et al. 2021)

Weighted Interval Score (WIS) — implementation notes:
    WIS(F, y) = (1 / (K + 0.5)) * [ 0.5 * |y - m|  +  sum_k (alpha_k/2 * IS_alpha_k(l_k, u_k, y)) ]

    where:
        m        = median forecast (quantile 0.5)
        K        = number of central interval pairs used
        alpha_k  = nominal miscoverage of the k-th interval (e.g. 0.10 for a 90% PI)
        IS_alpha = (u - l) + (2/alpha)*max(l - y, 0) + (2/alpha)*max(y - u, 0)

    Pairs used (any subset present in the submission files):
        (0.01, 0.99, alpha=0.02), (0.025, 0.975, 0.05), (0.05, 0.95, 0.10),
        (0.10, 0.90, 0.20), (0.15, 0.85, 0.30), (0.20, 0.80, 0.40),
        (0.25, 0.75, 0.50), (0.30, 0.70, 0.60), (0.35, 0.65, 0.70),
        (0.40, 0.60, 0.80), (0.45, 0.55, 0.90)

    Missing interval endpoints are skipped (graceful degradation).
    If only the median is present, WIS = |y - m| (K=0 edge case).

Usage:
    python numeric_eval_by_horizon.py [--forecast_dir ./datafiles] [--horizons 1,2,3] ...

Run from the project root directory.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

# Standard CDC FluSight symmetric interval pairs:
# (lower_quantile, upper_quantile, alpha=nominal_miscoverage)
# alpha = 1 - nominal_coverage  (e.g., 0.10 for a 90% PI)
CDC_INTERVAL_PAIRS = [
    (0.01, 0.99, 0.02),
    (0.025, 0.975, 0.05),
    (0.05, 0.95, 0.10),
    (0.10, 0.90, 0.20),
    (0.15, 0.85, 0.30),
    (0.20, 0.80, 0.40),
    (0.25, 0.75, 0.50),
    (0.30, 0.70, 0.60),
    (0.35, 0.65, 0.70),
    (0.40, 0.60, 0.80),
    (0.45, 0.55, 0.90),
]

# Helper functions
def zfill_loc(x) -> str:
    """Zero-pad a location FIPS code to 2 digits (e.g., '1' -> '01')."""
    return str(x).zfill(2)

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert column to datetime; unparseable values become NaT."""
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert column to float; unparseable values become NaN."""
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_horizon(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Derive forecast horizon (weeks ahead) as:
        horizon = floor((target_end_date - reference_date) / 7)

    Identical to the formula used in pred_vs_truth_epiweeks.py.
    The horizon column in the submission file is used directly if present,
    but recomputing from dates is more robust (guards against off-by-one).
    """
    delta_days = (pred["target_end_date"] - pred["reference_date"]).dt.days
    pred["horizon"] = (delta_days // 7).astype("Int64")
    return pred

# WIS
def interval_score(l, u, alpha, y):
    """
    Compute the Interval Score for a single prediction interval.

    IS_alpha(l, u, y) = (u - l)
                        + (2/alpha) * max(l - y, 0)   <- penalise undershoot
                        + (2/alpha) * max(y - u, 0)   <- penalise overshoot

    Parameters
    ----------
    l, u  : lower/upper bound (scalar or numpy array)
    alpha : miscoverage level (e.g. 0.10 for 90% PI)
    y     : observed truth

    Returns scalar or array of the same shape as inputs.
    """
    return (
        (u - l)
        + (2.0 / alpha) * np.maximum(l - y, 0.0)
        + (2.0 / alpha) * np.maximum(y - u, 0.0)
    )

def compute_wis_row(row_quantiles: dict, truth: float):
    """
    Compute the Weighted Interval Score for a single forecast-truth pair.

    Formula (Bracher et al. 2021, doi:10.1371/journal.pcbi.1009094):
        WIS = (1 / (K + 0.5)) * [ 0.5 * |y - m|  +  sum_{k=1}^{K} (alpha_k/2) * IS_alpha_k ]

    Parameters
    ----------
    row_quantiles : dict  {quantile_float -> predicted_value}
    truth         : observed value (float)

    Returns
    -------
    WIS (float) or None if the median is unavailable.
    """
    if 0.5 not in row_quantiles or np.isnan(row_quantiles[0.5]):
        return None

    median_pred = row_quantiles[0.5]
    median_term = 0.5 * abs(truth - median_pred)

    interval_terms = []
    for (lq, uq, alpha) in CDC_INTERVAL_PAIRS:
        l_val = row_quantiles.get(lq)
        u_val = row_quantiles.get(uq)
        if l_val is None or u_val is None:
            continue
        if np.isnan(l_val) or np.isnan(u_val):
            continue
        w_k = alpha / 2.0
        interval_terms.append(w_k * interval_score(l_val, u_val, alpha, truth))

    K = len(interval_terms)
    if K == 0:
        # Only median available: WIS degenerates to |y - m|
        # (K + 0.5 = 0.5, and the whole formula collapses to |y - m|)
        return median_term / 0.5

    return (1.0 / (K + 0.5)) * (median_term + sum(interval_terms))

# Data loading
def load_truth(truth_path: str):
    """
    Load CDC ground-truth and build a location code->name mapping.

    Deduplication:
        If as_of is present, keep the latest revision per (location, target_end_date).
        Otherwise average duplicates.  Same logic as pred_vs_truth_epiweeks.py.

    Returns
    -------
    truth      : cleaned DataFrame
    loc_mapping: dict {fips_str -> location_name_str}
    """
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    truth = pd.read_csv(truth_path)
    # Handle column name variants (space vs underscore)
    truth = truth.rename(columns={"as of": "as_of", "target end date": "target_end_date"})

    needed = {"target", "location_name", "target_end_date", "observation"}
    missing = needed - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV missing columns: {missing}. Found: {list(truth.columns)}")

    if "as_of" in truth.columns:
        truth = ensure_datetime(truth, "as_of")
    truth = ensure_datetime(truth, "target_end_date")
    truth = ensure_numeric(truth, "observation")

    if "location" in truth.columns:
        truth["location"] = truth["location"].apply(zfill_loc)

    # Build FIPS -> location_name lookup
    loc_mapping = {}
    if "location" in truth.columns:
        loc_mapping = (
            truth[["location", "location_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("location")["location_name"]
            .to_dict()
        )

    return truth, loc_mapping


def get_truth_lookup(truth: pd.DataFrame, target: str, loc_mapping: dict) -> pd.DataFrame:
    """
    Build a (location, target_end_date) -> observation lookup DataFrame for all locations.

    Deduplication: keep latest as_of revision per (location, target_end_date).
    Returns a DataFrame with columns [location, target_end_date, observation].
    """
    tgt_truth = truth[truth["target"] == target].copy()
    if tgt_truth.empty:
        return pd.DataFrame(columns=["location", "target_end_date", "observation"])

    if "as_of" in tgt_truth.columns:
        tgt_truth = tgt_truth.sort_values(["location", "target_end_date", "as_of"])
        tgt_truth = tgt_truth.drop_duplicates(subset=["location", "target_end_date"], keep="last")
    else:
        tgt_truth = (
            tgt_truth.groupby(["location", "target_end_date"], as_index=False)["observation"]
            .mean()
        )

    return tgt_truth[["location", "target_end_date", "observation"]].reset_index(drop=True)


def load_submissions(forecast_dir: str) -> pd.DataFrame:
    """
    Load all *-UM-DeepOutbreak.csv files from forecast_dir.

    Returns combined raw quantile DataFrame.
    """
    pattern = os.path.join(forecast_dir, "*-UM-DeepOutbreak.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No submission files matched: {pattern}")
    print(f"Found {len(files)} submission file(s) in {forecast_dir}")
    for f in files:
        print(f"  {os.path.basename(f)}")

    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    needed = {"reference_date", "target", "target_end_date", "location",
              "output_type", "output_type_id", "value"}
    missing = needed - set(pred.columns)
    if missing:
        raise ValueError(f"Submission CSV missing columns: {missing}")

    pred = ensure_datetime(pred, "reference_date")
    pred = ensure_datetime(pred, "target_end_date")
    pred = ensure_numeric(pred, "output_type_id")
    pred = ensure_numeric(pred, "value")
    pred["location"] = pred["location"].apply(zfill_loc)

    return pred


# Metric aggregation
def compute_metrics_for_group(df: pd.DataFrame) -> dict:
    """
    Aggregate evaluation metrics for a group of row-level forecast-truth pairs.

    Expects df to have columns: truth, pred_median, q_0.05, q_0.95,
                                 covered_90, abs_error, squared_error,
                                 width_90, wis_row.
    Rows with NaN inputs are excluded per metric (not globally).
    """
    n = len(df)

    mae_df = df.dropna(subset=["abs_error"])
    mae = mae_df["abs_error"].mean() if len(mae_df) > 0 else np.nan

    rmse_df = df.dropna(subset=["squared_error"])
    rmse = np.sqrt(rmse_df["squared_error"].mean()) if len(rmse_df) > 0 else np.nan

    cov_df = df.dropna(subset=["covered_90"])
    coverage_90 = cov_df["covered_90"].mean() if len(cov_df) > 0 else np.nan

    wid_df = df.dropna(subset=["width_90"])
    mean_width_90 = wid_df["width_90"].mean() if len(wid_df) > 0 else np.nan

    wis_df = df.dropna(subset=["wis_row"])
    wis = wis_df["wis_row"].mean() if len(wis_df) > 0 else np.nan

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "coverage_90": coverage_90,
        "mean_width_90": mean_width_90,
        "wis": wis,
    }

# Main
def main():
    parser = argparse.ArgumentParser(
        description="Numeric evaluation metrics for UM-DeepOutbreak flu forecasts by horizon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--forecast_dir", default=".",
        help="Folder containing *-UM-DeepOutbreak.csv submission files.",
    )
    parser.add_argument(
        "--truth_path", default=None,
        help="Path to cdc_datafiles.csv. Defaults to <forecast_dir>/cdc_datafiles.csv.",
    )
    parser.add_argument(
        "--output_summary", default="./plots/numeric_eval_summary.csv",
        help="Output path for the horizon-level summary CSV.",
    )
    parser.add_argument(
        "--output_detailed", default="./plots/numeric_eval_detailed.csv",
        help="Output path for the row-level detailed evaluation CSV.",
    )
    parser.add_argument(
        "--target", default="wk inc flu hosp",
        help="Forecast target string to evaluate.",
    )
    parser.add_argument(
        "--locations", default=None,
        help="Comma-separated FIPS codes to include (e.g. '01,06,26'). Default: all.",
    )
    parser.add_argument(
        "--horizons", default=None,
        help="Comma-separated horizons to evaluate (e.g. '1,2,3'). Default: all found.",
    )
    parser.add_argument(
        "--start_date", default=None,
        help="Earliest target_end_date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date", default=None,
        help="Latest target_end_date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--by_location", action="store_true",
        help="Also save a summary CSV grouped by horizon and location.",
    )
    args = parser.parse_args()

    # ---- Resolve paths ----
    truth_path = args.truth_path or os.path.join(args.forecast_dir, "cdc_datafiles.csv")
    print(f"\nTruth path   : {truth_path}")
    print(f"Forecast dir : {args.forecast_dir}")
    print(f"Target       : {args.target}")

    # Ensure output directories exist
    for out_path in [args.output_summary, args.output_detailed]:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)

    # 1) Load truth
    truth, loc_mapping = load_truth(truth_path)
    print(f"Truth rows loaded  : {len(truth)}")
    print(f"Locations in truth : {len(loc_mapping)}")

    # Build deduplicated truth lookup table (one row per location+target_end_date)
    truth_lookup_df = get_truth_lookup(truth, args.target, loc_mapping)

    # 2) Load and filter submissions
    pred = load_submissions(args.forecast_dir)
    print(f"Raw submission rows: {len(pred)}")

    pred = pred[(pred["target"] == args.target) & (pred["output_type"] == "quantile")].copy()
    print(f"After target+quantile filter: {len(pred)} rows")

    if args.locations:
        keep_locs = [zfill_loc(x.strip()) for x in args.locations.split(",")]
        pred = pred[pred["location"].isin(keep_locs)].copy()
        print(f"After location filter: {len(pred)} rows")

    if args.start_date:
        pred = pred[pred["target_end_date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        pred = pred[pred["target_end_date"] <= pd.to_datetime(args.end_date)].copy()

    # Recompute horizon from dates (robust to stale horizon column in CSVs)
    pred = compute_horizon(pred)

    if args.horizons:
        keep_horizons = [int(h.strip()) for h in args.horizons.split(",")]
        pred = pred[pred["horizon"].isin(keep_horizons)].copy()
        print(f"After horizon filter: {len(pred)} rows")

    all_locs = sorted(pred["location"].unique())
    all_horizons = sorted(pred["horizon"].dropna().unique())
    print(f"Locations : {len(all_locs)} | Horizons : {list(all_horizons)}")
    print(f"Quantiles available: {sorted(pred['output_type_id'].dropna().unique())}")

    # 3) Pivot quantiles wide: one row per (ref_date, ted, horizon, location)
    index_cols = ["reference_date", "target_end_date", "horizon", "location"]
    qwide = (
        pred.pivot_table(
            index=index_cols,
            columns="output_type_id",
            values="value",
            aggfunc="mean",     # handles rare duplicates gracefully
        )
        .reset_index()
    )
    qwide.columns.name = None  # remove the "output_type_id" axis name

    # Identify the quantile columns (floats like 0.01, 0.05, 0.5, 0.95, ...)
    quantile_cols = [c for c in qwide.columns if c not in index_cols]

    # 4) Attach observed truth values via a merge (avoids itertuples on
    #    float column names and is much faster than a row-wise lookup).
    qwide = qwide.merge(
        truth_lookup_df.rename(columns={"observation": "truth"}),
        on=["location", "target_end_date"],
        how="left",
    )

    # 5) Build detailed row-level evaluation table
    warn_missing_median = 0
    warn_missing_q05    = 0
    warn_missing_q95    = 0
    warn_missing_truth  = 0

    detailed_rows = []

    # Use iterrows() so float column names (0.5, 0.05 …) are accessed via row[col]
    # without the attribute-name mangling that itertuples() applies.
    for _, row in qwide.iterrows():
        # Build quantile dict {float_quantile -> float_value} for this instance
        row_q = {}
        for qc in quantile_cols:
            val = row[qc]
            if pd.notna(val):
                row_q[float(qc)] = float(val)

        truth_val  = row["truth"]
        has_truth  = pd.notna(truth_val)
        has_median = 0.5 in row_q
        has_q05    = 0.05 in row_q
        has_q95    = 0.95 in row_q

        if not has_truth:
            warn_missing_truth  += 1
        if not has_median:
            warn_missing_median += 1
        if not has_q05:
            warn_missing_q05    += 1
        if not has_q95:
            warn_missing_q95    += 1

        pred_median = row_q.get(0.5, np.nan)
        q05         = row_q.get(0.05, np.nan)
        q95         = row_q.get(0.95, np.nan)

        # Per-row metrics
        abs_error  = abs(truth_val - pred_median)       if (has_truth and has_median) else np.nan
        sq_error   = (truth_val - pred_median) ** 2     if (has_truth and has_median) else np.nan
        width_90   = (q95 - q05)                        if (has_q05 and has_q95) else np.nan
        covered_90 = (
            float(q05 <= truth_val <= q95)
            if (has_truth and has_q05 and has_q95)
            else np.nan
        )
        wis_row    = compute_wis_row(row_q, truth_val)  if (has_truth and has_median) else np.nan

        h = row["horizon"]
        detailed_rows.append({
            "reference_date" : row["reference_date"],
            "horizon"        : int(h) if pd.notna(h) else np.nan,
            "location"       : row["location"],
            "location_name"  : loc_mapping.get(row["location"], row["location"]),
            "target_end_date": row["target_end_date"],
            "truth"          : truth_val,
            "pred_median"    : pred_median,
            "q_0.05"         : q05,
            "q_0.95"         : q95,
            "covered_90"     : covered_90,
            "abs_error"      : abs_error,
            "squared_error"  : sq_error,
            "width_90"       : width_90,
            "wis_row"        : wis_row,
        })

    detailed = (
        pd.DataFrame(detailed_rows)
        .sort_values(["horizon", "location", "reference_date", "target_end_date"])
        .reset_index(drop=True)
    )

    # ---- Data quality report ----
    total = len(detailed)
    print(f"\n--- Data quality ({total} forecast instances after pivot) ---")
    print(f"  Missing observed truth : {warn_missing_truth:>6}  ({100*warn_missing_truth/max(total,1):.1f}%)")
    print(f"  Missing median (q0.50) : {warn_missing_median:>6}  ({100*warn_missing_median/max(total,1):.1f}%)")
    print(f"  Missing q0.05          : {warn_missing_q05:>6}  ({100*warn_missing_q05/max(total,1):.1f}%)")
    print(f"  Missing q0.95          : {warn_missing_q95:>6}  ({100*warn_missing_q95/max(total,1):.1f}%)")
    print(f"  WIS computed           : {detailed['wis_row'].notna().sum():>6}  / {total}")

    # Restrict summary metrics to rows where truth is observed
    with_truth = detailed[detailed["truth"].notna()].copy()
    n_skipped = total - len(with_truth)
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} instance(s) with no truth for summary metrics.")

    # 6) Horizon-level summary table
    summary_rows = []
    for h in sorted(with_truth["horizon"].dropna().unique()):
        grp = with_truth[with_truth["horizon"] == h]
        metrics = compute_metrics_for_group(grp)
        metrics["horizon"] = int(h)
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)[
        ["horizon", "n", "mae", "rmse", "coverage_90", "mean_width_90", "wis"]
    ].sort_values("horizon").reset_index(drop=True)

    # 7) Print summary table to terminal
    print("\n" + "=" * 72)
    print(f"NUMERIC EVALUATION — UM-DeepOutbreak  |  target: {args.target}")
    print("=" * 72)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("=" * 72)

    # 8) Save outputs
    summary.to_csv(args.output_summary, index=False)
    print(f"\nSummary saved  -> {args.output_summary}")

    detailed.to_csv(args.output_detailed, index=False)
    print(f"Detailed saved -> {args.output_detailed}")

    # 9) Optional by-location breakdown
    if args.by_location:
        loc_rows = []
        for h in sorted(with_truth["horizon"].dropna().unique()):
            for loc in sorted(with_truth["location"].unique()):
                grp = with_truth[
                    (with_truth["horizon"] == h) & (with_truth["location"] == loc)
                ]
                if grp.empty:
                    continue
                metrics = compute_metrics_for_group(grp)
                metrics["horizon"]       = int(h)
                metrics["location"]      = loc
                metrics["location_name"] = loc_mapping.get(loc, loc)
                loc_rows.append(metrics)

        loc_summary = pd.DataFrame(loc_rows)[
            ["horizon", "location", "location_name", "n",
             "mae", "rmse", "coverage_90", "mean_width_90", "wis"]
        ].sort_values(["horizon", "location"]).reset_index(drop=True)

        base, ext = os.path.splitext(args.output_summary)
        loc_out = f"{base}_by_location{ext}"
        loc_summary.to_csv(loc_out, index=False)
        print(f"By-location summary -> {loc_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
