"""
Apply epimodulation to RSV hospitalization forecasts.
Reference paper: https://www.pnas.org/doi/epdf/10.1073/pnas.2508575122 

This script applies a cumulative-prediction damping correction to base model
forecasts, estimates horizon-specific theta values on training data, and
evaluates whether the adjusted predictions improve MAE/RMSE on the test set.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = list(range(1, 9)) # folders 1–8
HORIZONS = [1, 2, 3, 4]
SENTINEL = -9 # missing ground-truth marker in true_<h>
THETA_GRID = np.linspace(0, 0.001, 500) # scale-adjusted: cumulative sums are O(100–30000)


# helper functions
def load_folder(folder_id: int) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, str(folder_id), f"{folder_id}.csv")
    df = pd.read_csv(path)
    df["folder"] = folder_id
    return df

def compute_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in HORIZONS:
        # making cumulative Ch 
        # ex. C3​=pred1​+pred2​+pred3​
        pred_cols = [f"pred_{j}" for j in range(1, h + 1)]
        df[f"C_{h}"] = df[pred_cols].sum(axis=1)
    return df

def apply_epimodulation(df: pd.DataFrame, thetas: dict) -> pd.DataFrame:
    """
    Apply epimodulation with given per-horizon thetas.
    Returns df with new columns epi_pred_<h> and ensures non-negative.
    """
    df = df.copy()
    for h in HORIZONS:
        # key equation from the paper: y~​h​ = y^​h​⋅exp(−θ⋅Ch​)
        raw = df[f"pred_{h}"].values
        C = df[f"C_{h}"].values
        damped = raw * np.exp(-thetas[h] * C)
        df[f"epi_pred_{h}"] = np.maximum(damped, 0.0)
    return df

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != SENTINEL
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != SENTINEL
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

def estimate_theta(train_df: pd.DataFrame, h: int) -> float:
    """
    Grid search over THETA_GRID to find theta_h that minimises MAE on train_df
    for horizon h.  Rows where true_h == SENTINEL are excluded.
    """
    pred = train_df[f"pred_{h}"].values
    C = train_df[f"C_{h}"].values
    true = train_df[f"true_{h}"].values
    mask = true != SENTINEL

    if mask.sum() == 0:
        return 0.0

    pred_m, C_m, true_m = pred[mask], C[mask], true[mask]

    best_theta, best_mae = 0.0, np.inf
    # put differet theta candidates => and then chose theta with the smallest MAE
    for theta in THETA_GRID:
        damped = np.maximum(pred_m * np.exp(-theta * C_m), 0.0)
        m = np.mean(np.abs(true_m - damped))
        if m < best_mae:
            best_mae = m
            best_theta = theta
    return best_theta


# main evaluation
def main():
    # 1. Load all folders
    frames = [load_folder(fid) for fid in FOLDERS]
    all_data = pd.concat(frames, ignore_index=True)

    all_data = compute_cumulative(all_data)

    train_df = all_data[all_data["is_test"] == False].copy()
    test_df  = all_data[all_data["is_test"] == True].copy()

    print(f"Loaded {len(FOLDERS)} folders | train rows: {len(train_df)} | test rows: {len(test_df)}")
    print()

    # 2. Estimate theta per horizon on non-test rows (no leakage into test set)
    thetas = {}
    for h in HORIZONS:
        thetas[h] = estimate_theta(train_df, h)
        print(f"  Estimated theta_{h} = {thetas[h]:.6f}")
    print()

    # 3. Apply epimodulation to the full dataset
    all_data = apply_epimodulation(all_data, thetas)
    train_df = all_data[all_data["is_test"] == False].copy()
    test_df  = all_data[all_data["is_test"] == True].copy()

    # 4. Compute metrics
    rows = []

    for split_name, df in [("train (non-test)", train_df), ("test", test_df)]:
        for h in HORIZONS:
            true = df[f"true_{h}"].values
            orig = df[f"pred_{h}"].values
            epi  = df[f"epi_pred_{h}"].values
            rows.append({
                "split": split_name,
                "horizon": h,
                "mae_original":      mae(true, orig),
                "mae_epimodulated":  mae(true, epi),
                "rmse_original":     rmse(true, orig),
                "rmse_epimodulated": rmse(true, epi),
                "theta": thetas[h],
            })

    metrics = pd.DataFrame(rows)
    metrics["mae_delta"]  = metrics["mae_epimodulated"]  - metrics["mae_original"]
    metrics["rmse_delta"] = metrics["rmse_epimodulated"] - metrics["rmse_original"]
    metrics["mae_improved"]  = metrics["mae_delta"]  < 0
    metrics["rmse_improved"] = metrics["rmse_delta"] < 0

    # 5. Print per-horizon summary
    print("=" * 70)
    print("Per-horizon metrics  (negative delta = improvement)")
    print("=" * 70)
    for _, r in metrics.iterrows():
        tag = "BETTER" if r["mae_improved"] else "worse "
        print(
            f"  [{r['split']:18s}] h={r['horizon']}  "
            f"MAE: {r['mae_original']:7.2f} -> {r['mae_epimodulated']:7.2f} "
            f"(delta={r['mae_delta']:+7.2f})  [{tag}]  "
            f"RMSE: {r['rmse_original']:7.2f} -> {r['rmse_epimodulated']:7.2f}"
        )
    print()

    # 6. Overall averages (across horizons)
    print("=" * 70)
    print("Overall averages (across all horizons)")
    print("=" * 70)
    for split_name in metrics["split"].unique():
        sub = metrics[metrics["split"] == split_name]
        print(f"\n  {split_name}:")
        print(f"    MAE  original:      {sub['mae_original'].mean():.4f}")
        print(f"    MAE  epimodulated:  {sub['mae_epimodulated'].mean():.4f}  "
              f"(delta: {sub['mae_delta'].mean():+.4f})")
        print(f"    RMSE original:      {sub['rmse_original'].mean():.4f}")
        print(f"    RMSE epimodulated:  {sub['rmse_epimodulated'].mean():.4f}  "
              f"(delta: {sub['rmse_delta'].mean():+.4f})")
        n_better = sub["mae_improved"].sum()
        print(f"    MAE improved on {n_better}/{len(sub)} horizons")
    print()

    # 7. Save a clean per-horizon summary CSV (test split only, easy to read)
    test_metrics = metrics[metrics["split"] == "test"].copy()
    summary = test_metrics[[
        "horizon", "theta",
        "mae_original", "mae_epimodulated", "mae_delta", "mae_improved",
        "rmse_original", "rmse_epimodulated", "rmse_delta", "rmse_improved",
    ]].reset_index(drop=True)
    summary_path = os.path.join(BASE_DIR, "epimodulation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Test-set summary saved to: {summary_path}")

    # 8. Save updated predictions
    save_cols = (
        ["folder", "region", "epiweek", "is_test", "target"]
        + [f"pred_{h}"     for h in HORIZONS]
        + [f"epi_pred_{h}" for h in HORIZONS]
        + [f"true_{h}"     for h in HORIZONS]
    )
    pred_path = os.path.join(BASE_DIR, "epimodulated_predictions.csv")
    all_data[save_cols].to_csv(pred_path, index=False)
    print(f"Updated predictions saved to: {pred_path}")

    # 9. Verdict
    test_sub = metrics[metrics["split"] == "test"]
    avg_delta = test_sub["mae_delta"].mean()
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if avg_delta < 0:
        print(f"  Epimodulation IMPROVED test MAE on average by {-avg_delta:.4f} "
              f"({test_sub['mae_improved'].sum()}/{len(test_sub)} horizons better)")
    elif avg_delta == 0:
        print("  Epimodulation had NO effect (theta=0 for all horizons).")
    else:
        print(f"  Epimodulation DID NOT improve test MAE "
              f"(average delta = +{avg_delta:.4f}; "
              f"{test_sub['mae_improved'].sum()}/{len(test_sub)} horizons better)")
    print(f"  Estimated thetas: { {h: round(v,6) for h,v in thetas.items()} }")


if __name__ == "__main__":
    main()