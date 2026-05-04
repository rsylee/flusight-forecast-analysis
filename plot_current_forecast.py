"""
plot_current_forecast.py

Generates a "current-week forecast" plot for UM-DeepOutbreak / CDC FluSight.

Plot layout
-----------
LEFT  (history)  : observed ground truth only, up to the last available week
RIGHT (forecast) : current-week forecast quantiles only (no historical predictions) 
SEPARATOR        : vertical dashed line at the forecast origin -- reference_date

Quantile bands shown (if the quantiles are present in the submission CSV):
  - 50%  PI  : 0.25 – 0.75
  - 80%  PI  : 0.10 – 0.90
  - 95%  PI  : 0.025 – 0.975
  - Median   : 0.50 (solid line)

Data sources
------------
- Ground truth  : cdc_datafiles.csv   (CDC-observed, latest revision per week) -- plotted on left side since they are true/real/verified data from CDC
- Forecast      : latest *-UM-DeepOutbreak.csv in forecast_dir (or --forecast_file) -- plotted on the right side as these are prediction

How the history / forecast split is determined
-----------------------------------------------
1. The latest submission CSV is selected (alphabetically last filename, which equals the most recent date because filenames are YYYY-MM-DD-…).
2. reference_date = max(reference_date) in that file = the forecast origin.
3. Truth is shown for all weeks whose target_end_date <= last observed truth date
   AND within the requested --history_weeks window.
4. Forecast is shown for all target_end_dates in the submission (horizons 0–4 by default).

Usage examples
--------------
# All locations, auto-detect latest submission, save to datafiles/
python3 plot_current_forecast.py

# Single location by FIPS code
python3 plot_current_forecast.py --locations 26

# Specific submission file
python3 plot_current_forecast.py --forecast_file 2026-02-21-UM-DeepOutbreak.csv

# Only horizons 1-4 (skip nowcast)
python3 plot_current_forecast.py --horizons 1,2,3,4

# Custom output directory
python3 plot_current_forecast.py --output_dir /tmp/fc_plots

Requirements:
pip install pandas numpy matplotlib epiweeks
"""

import argparse
import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epiweeks import Week

matplotlib.use("Agg")  # non-interactive backend – always saves, never pops a window

DEFAULT_TRUTH_PATH = "cdc_datafiles.csv"
DEFAULT_SUBMISSION_GLOB = "*-UM-DeepOutbreak.csv"
DEFAULT_TARGET = "wk inc flu hosp"
DEFAULT_HISTORY_WEEKS = 52          # weeks of truth to show before forecast origin
DEFAULT_HORIZONS = [0, 1, 2, 3, 4]  # which horizons to include in the forecast plot

# Quantile interval definitions  (low, high, nominal coverage, display label, alpha)
QUANTILE_BANDS = [
    (0.025, 0.975, "95% PI (0.025–0.975)", 0.12),
    (0.10,  0.90,  "80% PI (0.10–0.90)",   0.20),
    (0.25,  0.75,  "50% PI (0.25–0.75)",   0.30),
]
MEDIAN_QUANTILE = 0.50


# Helper fucntions
def zfill_loc(x) -> str: # -> str is just for explanation (doesn't do any work; just telling us that it will return a string)
    """Zero-pad location FIPS codes to 2 digits (e.g. 1 -> '01'); zfill = zero fill"""
    return str(x).zfill(2)

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column to datetime; invalid parses become NaT."""
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column to numeric; invalid parses become NaN."""
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def epiweek_label(dt: pd.Timestamp) -> str:
    """Convert a date to a CDC epiweek label like '2025-W46'."""
    if pd.isna(dt):
        return ""
    w = Week.fromdate(dt.date())
    return f"{w.year}-W{w.week:02d}"

def pick_latest_submission(forecast_dir: str, glob_pattern: str) -> str:
    """
    Return the path of the alphabetically-last submission CSV in forecast_dir.
    Since filenames are YYYY-MM-DD-…, alphabetical order == chronological order.
    forecast_dir = folder that has files in
    glob_pattern = pattern of the file that I wanna find
    """
    candidates = sorted(glob.glob(os.path.join(forecast_dir, glob_pattern)))
    if not candidates: # error if none exist
        raise FileNotFoundError(
            f"No submission files matched '{glob_pattern}' in '{forecast_dir}'.\n"
            "Pass --forecast_file to specify an explicit file."
        )
    latest = candidates[-1] # -1 is the last one (most recent one)
    print(f"[INFO] Auto-selected latest submission: {os.path.basename(latest)}")
    return latest


# Data loading
def load_truth(truth_path: str, target: str) -> pd.DataFrame:
    """
    Load and clean the CDC ground-truth CSV.

    Returns a DataFrame indexed by (location, target_end_date) with columns
    [observation, location_name].  Duplicate revisions are resolved by keeping
    the row with the latest as_of.
    """
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    truth = pd.read_csv(truth_path) # create DataFrame while reading csv (cdc_datafiles.csv)

    # Normalise column names (some exports use spaces)
    truth = truth.rename(columns={"as of": "as_of", "target end date": "target_end_date"})

    required = {"target", "location_name", "target_end_date", "observation"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV missing required columns: {missing}\nFound: {list(truth.columns)}")

    if "as_of" in truth.columns:
        truth = ensure_datetime(truth, "as_of")
    truth = ensure_datetime(truth, "target_end_date")
    truth = ensure_numeric(truth, "observation")

    # Keep only the target we care about
    truth = truth[truth["target"] == target].copy()

    # Zero-pad location codes so they match submission CSVs
    if "location" in truth.columns:
        truth["location"] = truth["location"].apply(zfill_loc)

    # Deduplicate: keep latest revision (max as_of) per (location, target_end_date)
    if "as_of" in truth.columns:
        truth = (
            truth
            .sort_values(["location", "target_end_date", "as_of"])
            .drop_duplicates(subset=["location", "target_end_date"], keep="last")
        )
    else:
        # Fallback: average duplicate observations
        truth = (
            truth
            .groupby(["location", "target_end_date", "location_name"], as_index=False)["observation"]
            .mean()
        )

    return truth.sort_values(["location", "target_end_date"]).reset_index(drop=True)


def load_forecast(forecast_file: str, target: str, horizons: list[int]) -> pd.DataFrame:
    """
    Load and clean a single submission CSV.

    Returns a long-format DataFrame with columns:
        reference_date, target_end_date, horizon, location, output_type_id, value
    filtered to the requested target and horizons.
    """
    if not os.path.exists(forecast_file):
        raise FileNotFoundError(f"Forecast file not found: {forecast_file}")

    pred = pd.read_csv(forecast_file) # our model prediction for future (UMDeppOutbreak.csv files for each week)

    required = {"reference_date", "target", "target_end_date", "location",
                "output_type", "output_type_id", "value"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Submission CSV missing required columns: {missing}\nFound: {list(pred.columns)}")

    # string to date
    pred = ensure_datetime(pred, "reference_date")
    pred = ensure_datetime(pred, "target_end_date")
    # string to int (date)
    pred = ensure_numeric(pred, "output_type_id")
    pred = ensure_numeric(pred, "value")
    pred["location"] = pred["location"].apply(zfill_loc)

    # Keep only quantile output for our target
    pred = pred[
        (pred["target"] == target) &
        (pred["output_type"] == "quantile")
    ].copy()

    if pred.empty:
        raise ValueError(
            f"No quantile rows found for target='{target}' in {forecast_file}.\n"
            f"Targets present: {pred['target'].unique().tolist()}"
        )

    # Compute horizon from dates (overrides any existing column, ensures consistency)
    delta_days = (pred["target_end_date"] - pred["reference_date"]).dt.days
    pred["horizon"] = (delta_days // 7).astype("Int64")

    pred = pred[pred["horizon"].isin(horizons)].copy()
    if pred.empty:
        raise ValueError(
            f"No rows remain after filtering horizons={horizons}.\n"
            f"Horizons present: {sorted(pred['horizon'].dropna().unique().tolist())}"
        )

    # Use the single reference_date (latest in the file, in case file has multiple)
    ref_date = pred["reference_date"].max()
    print(f"[INFO] Forecast reference_date (origin): {ref_date.date()}")
    pred = pred[pred["reference_date"] == ref_date].copy()

    return pred, ref_date


# Plotting
def plot_location_forecast(
    loc_code: str,
    loc_name: str,
    truth_loc: pd.DataFrame,      # truth filtered to this location, indexed by target_end_date
    pred_loc: pd.DataFrame,       # forecast filtered to this location (long format)
    ref_date: pd.Timestamp,
    history_weeks: int,
    output_dir: str,
    target: str,
) -> str:
    """
    Generate and save the current-forecast plot for one location.
    Returns the saved file path.
    """
    # 1) Truth: only up to last observed week, within history window 
    last_truth_date = truth_loc.index.max()
    history_cutoff = ref_date - pd.Timedelta(weeks=history_weeks)

    truth_plot = truth_loc[
        (truth_loc.index >= history_cutoff) &
        (truth_loc.index <= last_truth_date)
    ].copy()

    if truth_plot.empty:
        print(f"  [WARN] No truth data for {loc_name} ({loc_code}); skipping.")
        return None

    # 2) Forecast: pivot quantiles wide
    q = (
        pred_loc
        .pivot_table(
            index="target_end_date",
            columns="output_type_id",
            values="value",
            aggfunc="mean",
        )
        .sort_index()
    )

    if MEDIAN_QUANTILE not in q.columns:
        print(f"  [WARN] Median quantile ({MEDIAN_QUANTILE}) missing for {loc_name}; skipping.")
        return None

    def get_q(qtile):
        """Return quantile series, or all-NaN if not available."""
        if qtile in q.columns:
            return q[qtile]
        return pd.Series(np.nan, index=q.index)

    # 3) Build figure (fig = canvas; ax = an area for drawing a graph)
    fig, ax = plt.subplots(figsize=(13, 5))

    # Historical ground truth (left side)
    ax.plot(
        truth_plot.index, # x-axis
        truth_plot["observation"], # y-axis
        color="#1f77b4", # mpl blue
        linewidth=1.8,
        marker="o",
        markersize=4,
        alpha=0.85,
        label="Observed (ground truth)",
        zorder=3,
    )

    # Optional: draw a thin connector from the last truth point to the first
    # forecast point so the visual story is continuous.
    first_fc_date = q.index.min()
    last_truth_val = truth_plot["observation"].iloc[-1]
    median_at_first = get_q(MEDIAN_QUANTILE).iloc[0] if not get_q(MEDIAN_QUANTILE).empty else np.nan
    if pd.notna(last_truth_val) and pd.notna(median_at_first):
        ax.plot(
            [truth_plot.index[-1], first_fc_date],
            [last_truth_val, median_at_first],
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.6,
            zorder=2,
        )

    # Forecast quantile bands (right side, widest first → narrowest on top)
    available_bands = []
    for (lo, hi, label, alpha) in QUANTILE_BANDS:
        lo_s = get_q(lo)
        hi_s = get_q(hi)
        if lo_s.notna().any() and hi_s.notna().any():
            ax.fill_between(
                q.index,
                lo_s.values,
                hi_s.values,
                alpha=alpha,
                color="#ff7f0e", # mpl orange
                label=label,
                zorder=1,
            )
            available_bands.append(label)
        else:
            print(f"  [INFO] Quantiles {lo}–{hi} not available; skipping that band.")

    # Forecast median line (from reference date)
    ax.plot(
        q.index,
        get_q(MEDIAN_QUANTILE).values,
        color="#ff7f0e",
        linewidth=2.2,
        marker="x",
        markersize=6,
        linestyle="-",
        label="Forecast median (q0.50)",
        zorder=4,
    )

    # Vertical separator at forecast origin
    ax.axvline(
        ref_date,
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label=f"Forecast origin ({ref_date.date()})",
        zorder=5,
    )

    # 4) Axes cosmetics
    # Build unified x-tick positions: evenly spaced across the full date range
    all_dates = truth_plot.index.tolist() + q.index.tolist()
    all_dates_sorted = sorted(set(all_dates))
    step = max(1, len(all_dates_sorted) // 14)  # at most ~14 ticks
    tick_dates = all_dates_sorted[::step]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(
        [epiweek_label(pd.Timestamp(d)) for d in tick_dates],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_xlabel("Epiweek (week-ending date)", fontsize=10)
    ax.set_ylabel("Weekly incident flu hospitalizations", fontsize=10)
    ax.set_title(
        f"{target} — {loc_name} (loc={loc_code})\n"
        f"Truth through {epiweek_label(last_truth_date)}  |  "
        f"Forecast origin {epiweek_label(ref_date)}  |  "
        f"Horizons shown: {DEFAULT_HORIZONS}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_xlim(
        left=history_cutoff - pd.Timedelta(days=7),
        right=q.index.max() + pd.Timedelta(days=7),
    )

    # Ensure y-axis starts at 0 (hospitalization counts can't be negative)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=max(0, ymin), top=ymax * 1.05)

    plt.tight_layout()

    # 5) Save
    os.makedirs(output_dir, exist_ok=True)
    safe_name = loc_name.replace(" ", "_").replace("/", "_")
    fname = f"current_forecast_{ref_date.date()}_{loc_code}_{safe_name}.png" # save it as png file
    outpath = os.path.join(output_dir, fname)
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

# CLI / main
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot current-week UM-DeepOutbreak forecast: truth (left) + forecast quantiles (right).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--forecast_dir",
        default=".",
        help="Directory containing *-UM-DeepOutbreak.csv files. Default: current directory.",
    )
    p.add_argument(
        "--forecast_file",
        default=None,
        help="Explicit submission CSV to use. If omitted, the latest file in forecast_dir is chosen.",
    )
    p.add_argument(
        "--truth_path",
        default=DEFAULT_TRUTH_PATH,
        help=f"Path to CDC ground-truth CSV. Default: {DEFAULT_TRUTH_PATH}",
    )
    p.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save plot PNG files. Default: . (datafiles/ folder).",
    )
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Forecast target string. Default: '{DEFAULT_TARGET}'",
    )
    p.add_argument(
        "--locations",
        default=None,
        help="Comma-separated FIPS codes to plot (e.g. '26,17,US'). Default: all locations.",
    )
    p.add_argument(
        "--horizons",
        default=",".join(map(str, DEFAULT_HORIZONS)),
        help=f"Comma-separated horizons to include. Default: '{','.join(map(str, DEFAULT_HORIZONS))}'",
    )
    p.add_argument(
        "--history_weeks",
        type=int,
        default=DEFAULT_HISTORY_WEEKS,
        help=f"Number of historical weeks to show before forecast origin. Default: {DEFAULT_HISTORY_WEEKS}",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("cwd:", os.getcwd())

    # Parse horizons
    try:
        horizons = [int(h.strip()) for h in args.horizons.split(",")]
    except ValueError:
        sys.exit(f"[ERROR] --horizons must be comma-separated integers, got: {args.horizons!r}")

    # 1) Load truth
    print(f"\n[INFO] Loading truth from: {args.truth_path}")
    truth = load_truth(args.truth_path, args.target)

    # Build location-code -> name mapping (needed to join truth to forecast)
    loc_map = (
        truth[["location", "location_name"]]
        .dropna()
        .drop_duplicates()
        .set_index("location")["location_name"]
        .to_dict()
    )

    # 2) Load forecast
    if args.forecast_file:
        fc_path = args.forecast_file
        if not os.path.isabs(fc_path):
            fc_path = os.path.join(args.forecast_dir, fc_path)
        print(f"[INFO] Using specified forecast file: {fc_path}")
    else:
        fc_path = pick_latest_submission(args.forecast_dir, DEFAULT_SUBMISSION_GLOB)

    print(f"[INFO] Loading forecast from: {os.path.basename(fc_path)}")
    pred, ref_date = load_forecast(fc_path, args.target, horizons)

    # 3) Determine locations to plot
    if args.locations:
        requested = [zfill_loc(l.strip()) for l in args.locations.split(",")]
        all_locs = [l for l in requested if l in pred["location"].values]
        missing_locs = set(requested) - set(all_locs)
        if missing_locs:
            print(f"[WARN] Locations not found in forecast: {missing_locs}")
    else:
        all_locs = sorted(pred["location"].unique())

    print(f"[INFO] Plotting {len(all_locs)} location(s) → output_dir: {os.path.abspath(args.output_dir)}\n")

    # 4) Plot each location
    saved_paths = []
    for loc_code in all_locs:
        loc_name = loc_map.get(loc_code, loc_code)
        print(f"  Processing: {loc_name} ({loc_code})")

        # Truth for this location (indexed by target_end_date)
        truth_loc = truth[truth["location"] == loc_code].copy()
        if truth_loc.empty:
            print(f"    [WARN] No truth rows for location {loc_code}; skipping.")
            continue
        truth_loc = truth_loc.sort_values("target_end_date").set_index("target_end_date")

        # Forecast for this location
        pred_loc = pred[pred["location"] == loc_code].copy()
        if pred_loc.empty:
            print(f"    [WARN] No forecast rows for location {loc_code}; skipping.")
            continue

        outpath = plot_location_forecast(
            loc_code=loc_code,
            loc_name=loc_name,
            truth_loc=truth_loc,
            pred_loc=pred_loc,
            ref_date=ref_date,
            history_weeks=args.history_weeks,
            output_dir=args.output_dir,
            target=args.target,
        )
        if outpath:
            print(f"    Saved → {outpath}")
            saved_paths.append(outpath)

    # 5) Summary
    print(f"\n[DONE] {len(saved_paths)} plot(s) saved.")
    for p in saved_paths:
        print(f"  {p}")

if __name__ == "__main__":
    main()