"""
Evaluation plots:
- One figure per horizon (1–4 weeks ahead)
- X-axis is Week Number (1..N) rather than dates
- Two panels:
  (Top) Coverage: target coverage line, realized coverage line, running coverage
  (Bottom) Ground truth vs predicted median + 90% prediction interval
           + red dots for missed points (truth outside PI)

Inputs:
- CDC truth: cdc_datafiles.csv  (needs columns: target, location_name, target_end_date, observation, optionally as_of)
- Submissions: CSVs generated in 2026 (ex. 2026-01-31-UM-DeepOutbreak.csv, 2026-02-07-UM-DeepOutbreak.csv, ...)

How horizon is computed:
- horizon = floor((target_end_date - reference_date) / 7)
- Then we keep only horizons 1..4

Output:
- Saves plots to ./plots/
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRUTH_PATH = "cdc_datafiles.csv"
SUBMISSION_GLOB = "*-UM-DeepOutbreak.csv"
TARGET = "wk inc flu hosp"

# (only tried Alabama for now)
LOCATION_CODE = "01"      # submission uses code
LOCATION_NAME = "Alabama" # truth uses name

HORIZONS = [1, 2, 3, 4]

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

# helper functions
def zfill_loc(x) -> str:
    return str(x).zfill(2)

def ensure_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def ensure_numeric(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_horizon(pred: pd.DataFrame) -> pd.DataFrame:
    delta_days = (pred["target_end_date"] - pred["reference_date"]).dt.days
    pred["horizon"] = (delta_days // 7).astype("Int64")
    pred = pred[pred["horizon"].isin([1, 2, 3, 4])].copy()
    return pred

def choose_pi_bounds_from_columns(cols: set):
    if 0.05 in cols and 0.95 in cols:
        return 0.05, 0.95, 0.90, "90% Prediction Interval (0.05–0.95)"
    if 0.10 in cols and 0.90 in cols:
        return 0.10, 0.90, 0.80, "80% Prediction Interval (0.10–0.90)"
    if 0.025 in cols and 0.975 in cols:
        return 0.025, 0.975, 0.95, "95% Prediction Interval (0.025–0.975)"
    return None, None, None, "No PI quantiles available"

# main
def main():
    print("cwd:", os.getcwd())

    # 1) Load truth (ground truth)
    if not os.path.exists(TRUTH_PATH):
        raise FileNotFoundError(f"Truth file not found: {TRUTH_PATH}")

    truth = pd.read_csv(TRUTH_PATH)
    truth = truth.rename(columns={"as of": "as_of", "target end date": "target_end_date"})

    needed_truth_cols = {"target", "location_name", "target_end_date", "observation"}
    missing_truth = needed_truth_cols - set(truth.columns)
    if missing_truth:
        raise ValueError(f"Truth CSV missing columns: {missing_truth}\nColumns found: {list(truth.columns)}")

    if "as_of" in truth.columns:
        truth = ensure_datetime(truth, "as_of")
    truth = ensure_datetime(truth, "target_end_date")
    truth = ensure_numeric(truth, "observation")

    # filter truth to target/location
    snap_truth = truth[
        (truth["target"] == TARGET) &
        (truth["location_name"] == LOCATION_NAME)
    ].copy()

    # deduplicate by keeping latest revision per target_end_date (max as_of)
    if "as_of" in snap_truth.columns:
        snap_truth = snap_truth.sort_values(["target_end_date", "as_of"])
        snap_truth = snap_truth.drop_duplicates(subset=["target_end_date"], keep="last")
    else:
        snap_truth = snap_truth.groupby("target_end_date", as_index=False)["observation"].mean()

    snap_truth = snap_truth.sort_values("target_end_date").set_index("target_end_date")
    print("Truth rows:", len(snap_truth))

    # 2) Load submissions (multiple files)
    files = sorted(glob.glob(SUBMISSION_GLOB))
    if not files:
        raise FileNotFoundError(f"No submission files matched: {SUBMISSION_GLOB}")

    print("Submission files found:", len(files))
    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    needed_pred_cols = {"reference_date", "target", "target_end_date", "location",
                        "output_type", "output_type_id", "value"}
    missing_pred = needed_pred_cols - set(pred.columns)
    if missing_pred:
        raise ValueError(f"Submission CSV missing columns: {missing_pred}\nColumns found: {list(pred.columns)}")

    pred = ensure_datetime(pred, "reference_date")
    pred = ensure_datetime(pred, "target_end_date")
    pred = ensure_numeric(pred, "output_type_id")
    pred = ensure_numeric(pred, "value")
    pred["location"] = pred["location"].apply(zfill_loc)

    # filter to our target/location/quantile only
    pred = pred[
        (pred["target"] == TARGET) &
        (pred["location"] == zfill_loc(LOCATION_CODE)) &
        (pred["output_type"] == "quantile")
    ].copy()

    if pred.empty:
        raise ValueError("After filtering, pred is empty. Check TARGET/LOCATION_CODE/output_type.")

    # compute horizon and keep only 1..4 (prevents weird horizons like 123)
    pred = compute_horizon(pred)

    print("Unique horizons after filtering:", sorted(pred["horizon"].dropna().unique()))
    print("Quantiles available:", sorted(pred["output_type_id"].dropna().unique()))

    # 3) Pivot quantiles wide: one row per forecast instance
    qwide = (
        pred.pivot_table(
            index=["reference_date", "target_end_date", "horizon"],
            columns="output_type_id",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
        .sort_values(["horizon", "target_end_date", "reference_date"])
    )

    # merge truth by target_end_date
    qwide = qwide.merge(
        snap_truth[["observation"]].reset_index(),
        on="target_end_date",
        how="left"
    )

    # 4) Plot per horizon
    for h in HORIZONS:
        dfh = qwide[qwide["horizon"] == h].copy()

        # sort by time and create "Week Number" (1..N)
        dfh = dfh.sort_values("target_end_date").reset_index(drop=True)
        dfh["week_number"] = np.arange(1, len(dfh) + 1)

        if dfh.empty:
            print(f"[h={h}] No data; skipping.")
            continue

        if 0.5 not in dfh.columns:
            print(f"[h={h}] Missing median (0.5) quantile; skipping.")
            continue

        # choose interval bounds (prefer 90%)
        cols = set(dfh.columns)
        low_q, high_q, target_cov, pi_label = choose_pi_bounds_from_columns(cols)

        x = dfh["week_number"]
        y_true = dfh["observation"]
        y_med = dfh[0.5]

        # coverage computation (only where truth exists)
        if low_q is not None and high_q is not None:
            inside = (y_true >= dfh[low_q]) & (y_true <= dfh[high_q])
            inside_num = inside.astype(float)
            inside_num[y_true.isna()] = np.nan

            running_cov = inside_num.expanding(min_periods=1).mean()
            realized_cov = inside_num.mean(skipna=True)

            missed = (~inside) & (~y_true.isna())
        else:
            running_cov = None
            realized_cov = None
            missed = pd.Series([False] * len(dfh))

        # 2-panel figure
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [1, 3]},
            sharex=True
        )

        # top: coverage
        ax_top.set_ylabel("Coverage")
        ax_top.set_ylim(0, 1.0)

        if running_cov is not None:
            # target coverage: black dashed
            ax_top.axhline(
                target_cov,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Target Coverage"
            )

            # realized coverage: orange dashed
            ax_top.axhline(
                realized_cov,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                label="Realized Coverage"
            )

            # running coverage: blue X line
            ax_top.plot(
                x,
                running_cov,
                color="blue",
                marker="x",
                label="Running Coverage"
            )
            ax_top.legend(loc="lower left")
        else:
            ax_top.text(0.01, 0.5, "No PI quantiles available for coverage",
                        transform=ax_top.transAxes)

        # bottom: truth vs prediction + PI
        ax_bot.set_xlabel("Week Number")
        ax_bot.set_ylabel(TARGET)

        ax_bot.plot(x, y_true, label="Ground Truth")
        ax_bot.plot(x, y_med, linestyle="--", label="Prediction")

        if low_q is not None and high_q is not None:
            ax_bot.fill_between(x, dfh[low_q].values, dfh[high_q].values, alpha=0.2, label=pi_label)

        # missed points (truth outside PI)
        if missed.any():
            ax_bot.scatter(x[missed], y_true[missed], color="red", label="Missed Points")

        ax_bot.legend(loc="upper left")

        # title with date range (optional but helpful)
        start_date = dfh["target_end_date"].min().date()
        end_date = dfh["target_end_date"].max().date()
        fig.suptitle(
            f"{TARGET} — {LOCATION_NAME} (location={LOCATION_CODE}) | Horizon={h} week ahead | {start_date} to {end_date} | Interval: {pi_label}",
            y=0.98
        )

        plt.tight_layout()
        outpath = os.path.join(OUTDIR, f"pred_vs_truth_h{h}.png")
        plt.savefig(outpath, dpi=200)
        plt.show()

        print(f"[h={h}] Saved: {outpath}")

    print("Done.")


if __name__ == "__main__":
    main()