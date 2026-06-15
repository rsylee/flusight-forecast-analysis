"""
Generates evaluation plots for ALL locations.

What this script does:
1) Loads CDC ground-truth ("cdc_datafiles.csv") and keeps the latest revision per week (max as_of).
2) Loads ALL UM-DeepOutbreak submission CSVs (2025 + 2026), filters to:
   - target = wk inc flu hosp
   - output_type = quantile
3) Loops over every location found in submissions and produces:
   A) Per-horizon evaluation plots (h=1..4):
      - X axis uses ACTUAL target_end_date (week-ending date)
      - X tick labels show EPIWEEK (YYYY-Www)
      - Top panel: coverage lines computed ONLY for weeks where observation exists
      - Bottom panel: truth vs predicted median + prediction interval; missed points shown
   B) Optional: Per-reference_date plot (horizons 1..4 together per submission week)

Outputs:
- Saves plots into ./plots/by_location/{location_name}/
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epiweeks import Week

TRUTH_PATH = "cdc_datafiles.csv"
SUBMISSION_GLOB = "202[56]-??-??-UM-DeepOutbreak.csv"
TARGET = "wk inc flu hosp"
KEEP_HORIZONS = [0, 1, 2, 3, 4]
PLOT_HORIZONS = [1, 2, 3, 4]

# plot output base directory (per-location subfolders created automatically)
OUTDIR_BASE = os.path.join("plots", "by_location")

# optional additional plot: show horizons 1..4 together per reference_date
MAKE_PER_REFERENCE_PLOT = True
MAX_REFERENCE_PLOTS = 12
# optional: limit to a time window (None = no limit)
START_DATE = None
END_DATE = None

# helper functions
def zfill_loc(x) -> str:
    return str(x).zfill(2)

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_horizon(pred: pd.DataFrame) -> pd.DataFrame:
    delta_days = (pred["target_end_date"] - pred["reference_date"]).dt.days
    pred["horizon"] = (delta_days // 7).astype("Int64")
    return pred

def epiweek_label(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return ""
    w = Week.fromdate(dt.date())
    return f"{w.year}-W{w.week:02d}"

def choose_pi_bounds_from_columns(cols: set):
    if 0.05 in cols and 0.95 in cols:
        return 0.05, 0.95, 0.90, "90% Prediction Interval (0.05–0.95)"
    if 0.10 in cols and 0.90 in cols:
        return 0.10, 0.90, 0.80, "80% Prediction Interval (0.10–0.90)"
    if 0.025 in cols and 0.975 in cols:
        return 0.025, 0.975, 0.95, "95% Prediction Interval (0.025–0.975)"
    return None, None, None, "No PI quantiles available"

def apply_date_window(df: pd.DataFrame, col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        df = df[df[col] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df[col] <= pd.to_datetime(end)]
    return df


# plot helpers (operate on a single location's data)
def plot_location(qwide, snap_truth, loc_code, loc_name, outdir):
    os.makedirs(outdir, exist_ok=True)

    # 4A) Per-horizon plots
    for h in PLOT_HORIZONS:
        dfh = qwide[qwide["horizon"] == h].copy()
        dfh = dfh.sort_values("target_end_date").reset_index(drop=True)

        if dfh.empty:
            print(f"  [h={h}] No data; skipping.")
            continue
        if 0.5 not in dfh.columns:
            print(f"  [h={h}] Missing median quantile; skipping.")
            continue

        low_q, high_q, target_cov, pi_label = choose_pi_bounds_from_columns(set(dfh.columns))

        x_all = dfh["target_end_date"]
        y_true_all = dfh["observation"]
        y_med_all = dfh[0.5]

        has_truth = ~dfh["observation"].isna()
        df_cov = dfh[has_truth].copy()

        if low_q is not None and high_q is not None and not df_cov.empty:
            inside = (df_cov["observation"] >= df_cov[low_q]) & (df_cov["observation"] <= df_cov[high_q])
            inside_num = inside.astype(float)
            running_cov = inside_num.expanding(min_periods=1).mean()
            realized_cov = inside_num.mean()
            missed = ~inside
        else:
            running_cov = None
            realized_cov = None
            missed = pd.Series([], dtype=bool)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={"height_ratios": [1, 3]},
            sharex=True
        )

        ax_top.set_ylabel("Coverage")
        ax_top.set_ylim(0, 1.0)
        if running_cov is not None:
            ax_top.axhline(target_cov, color="black", linestyle="--", linewidth=1.5, label="Target Coverage")
            ax_top.axhline(realized_cov, color="orange", linestyle="--", linewidth=1.5, label="Realized Coverage")
            ax_top.plot(df_cov["target_end_date"], running_cov, color="blue", marker="x", label="Running Coverage")
            ax_top.legend(loc="lower left")
        else:
            ax_top.text(0.01, 0.5, "Coverage only available when PI quantiles AND observation exist",
                        transform=ax_top.transAxes)

        ax_bot.set_xlabel("Epiweek (week-ending date)")
        ax_bot.set_ylabel(TARGET)
        ax_bot.plot(x_all, y_true_all, label="Ground Truth")
        ax_bot.plot(x_all, y_med_all, linestyle="--", label="Prediction (median)")
        if low_q is not None and high_q is not None:
            ax_bot.fill_between(x_all, dfh[low_q].values, dfh[high_q].values, alpha=0.2, label=pi_label)
        if not missed.empty and missed.any():
            ax_bot.scatter(df_cov["target_end_date"][missed], df_cov["observation"][missed],
                           color="red", label="Missed Points")
        ax_bot.legend(loc="upper left")

        step = max(1, len(dfh) // 10)
        tick_idx = np.arange(0, len(dfh), step)
        ax_bot.set_xticks(dfh["target_end_date"].iloc[tick_idx])
        ax_bot.set_xticklabels(dfh["epiweek"].iloc[tick_idx], rotation=45, ha="right")

        start_date = dfh["target_end_date"].min().date()
        end_date = dfh["target_end_date"].max().date()
        fig.suptitle(
            f"{TARGET} — {loc_name} (loc={loc_code}) | Horizon={h} week ahead | "
            f"{start_date} to {end_date} | {pi_label}",
            y=0.98
        )
        plt.tight_layout()
        outpath = os.path.join(outdir, f"pred_vs_truth_h{h}_epiweek.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  [h={h}] Saved: {outpath}")

    # 4B) Per-reference-date plots
    if MAKE_PER_REFERENCE_PLOT:
        refs = sorted(qwide["reference_date"].dropna().unique())[-MAX_REFERENCE_PLOTS:]
        for ref in refs:
            dfr = qwide[(qwide["reference_date"] == ref) & qwide["horizon"].isin([1, 2, 3, 4])].copy()
            dfr = dfr.sort_values("horizon")
            if dfr.empty or 0.5 not in dfr.columns:
                continue

            low_q, high_q, _, pi_label = choose_pi_bounds_from_columns(set(dfr.columns))
            xh = dfr["horizon"].astype(int)
            y_true = dfr["observation"]
            y_med = dfr[0.5]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(xh, y_true, marker="o", label="Ground Truth")
            ax.plot(xh, y_med, marker="x", linestyle="--", label="Prediction (median)")
            if low_q is not None and high_q is not None:
                ax.fill_between(xh, dfr[low_q].values, dfr[high_q].values, alpha=0.2, label=pi_label)
                has_t = ~y_true.isna()
                missed = has_t & ((y_true < dfr[low_q]) | (y_true > dfr[high_q]))
                if missed.any():
                    ax.scatter(xh[missed], y_true[missed], color="red", label="Missed Points")

            ax.set_xlabel("Horizon (weeks ahead)")
            ax.set_ylabel(TARGET)
            ref_str = pd.to_datetime(ref).date()
            ax.set_title(f"{TARGET} — {loc_name} | reference_date={ref_str} (1..4 horizons)")
            ax.set_xticks([1, 2, 3, 4])
            ax.legend(loc="upper left")
            plt.tight_layout()
            outpath = os.path.join(outdir, f"refdate_{ref_str}_h1to4.png")
            plt.savefig(outpath, dpi=150)
            plt.close()
            print(f"  [ref={ref_str}] Saved: {outpath}")

# main
def main():
    print("cwd:", os.getcwd())

    # 1) Load & clean truth
    if not os.path.exists(TRUTH_PATH):
        raise FileNotFoundError(f"Truth file not found: {TRUTH_PATH}")

    truth = pd.read_csv(TRUTH_PATH)
    truth = truth.rename(columns={"as of": "as_of", "target end date": "target_end_date"})

    needed_truth_cols = {"target", "location_name", "target_end_date", "observation"}
    missing_truth = needed_truth_cols - set(truth.columns)
    if missing_truth:
        raise ValueError(f"Truth CSV missing columns: {missing_truth}")

    if "as_of" in truth.columns:
        truth = ensure_datetime(truth, "as_of")
    truth = ensure_datetime(truth, "target_end_date")
    truth = ensure_numeric(truth, "observation")

    # build location code -> name mapping from truth
    loc_mapping = (
        truth[["location", "location_name"]]
        .dropna()
        .drop_duplicates()
        .assign(location=lambda df: df["location"].apply(zfill_loc))
        .set_index("location")["location_name"]
        .to_dict()
    )

    # 2) Load & clean all submissions
    files = sorted(glob.glob(SUBMISSION_GLOB))
    if not files:
        raise FileNotFoundError(f"No submission files matched: {SUBMISSION_GLOB}")
    print(f"Submission files found: {len(files)}")

    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    needed_pred_cols = {"reference_date", "target", "target_end_date", "location",
                        "output_type", "output_type_id", "value"}
    missing_pred = needed_pred_cols - set(pred.columns)
    if missing_pred:
        raise ValueError(f"Submission CSV missing columns: {missing_pred}")

    pred = ensure_datetime(pred, "reference_date")
    pred = ensure_datetime(pred, "target_end_date")
    pred = ensure_numeric(pred, "output_type_id")
    pred = ensure_numeric(pred, "value")
    pred["location"] = pred["location"].apply(zfill_loc)

    pred = pred[(pred["target"] == TARGET) & (pred["output_type"] == "quantile")].copy()
    pred = apply_date_window(pred, "target_end_date", START_DATE, END_DATE)
    pred = compute_horizon(pred)
    pred = pred[pred["horizon"].isin(KEEP_HORIZONS)].copy()

    all_loc_codes = sorted(pred["location"].unique())
    print(f"Locations to process: {len(all_loc_codes)}")

    # 3) Loop over locations
    for loc_code in all_loc_codes:
        loc_name = loc_mapping.get(loc_code, loc_code)
        print(f"\n=== {loc_name} ({loc_code}) ===")

        # filter truth for this location
        snap_truth = truth[
            (truth["target"] == TARGET) &
            (truth["location_name"] == loc_name)
        ].copy()

        if snap_truth.empty:
            print(f"  No truth data found; skipping.")
            continue

        if "as_of" in snap_truth.columns:
            snap_truth = snap_truth.sort_values(["target_end_date", "as_of"])
            snap_truth = snap_truth.drop_duplicates(subset=["target_end_date"], keep="last")
        else:
            snap_truth = snap_truth.groupby("target_end_date", as_index=False)["observation"].mean()

        snap_truth = snap_truth.sort_values("target_end_date").set_index("target_end_date")
        snap_truth = apply_date_window(snap_truth.reset_index(), "target_end_date", START_DATE, END_DATE).set_index("target_end_date")

        # filter predictions for this location
        pred_loc = pred[pred["location"] == loc_code].copy()
        if pred_loc.empty:
            print(f"  No prediction data; skipping.")
            continue

        # pivot quantiles wide
        qwide = (
            pred_loc.pivot_table(
                index=["reference_date", "target_end_date", "horizon"],
                columns="output_type_id",
                values="value",
                aggfunc="mean"
            )
            .reset_index()
            .sort_values(["horizon", "target_end_date", "reference_date"])
        )

        qwide = qwide.merge(
            snap_truth[["observation"]].reset_index(),
            on="target_end_date",
            how="left"
        )
        qwide["epiweek"] = qwide["target_end_date"].apply(epiweek_label)

        # safe folder name (replace spaces/slashes)
        safe_name = loc_name.replace(" ", "_").replace("/", "_")
        outdir = os.path.join(OUTDIR_BASE, f"{loc_code}_{safe_name}")

        plot_location(qwide, snap_truth, loc_code, loc_name, outdir)

    print("\nAll done.")


if __name__ == "__main__":
    main()