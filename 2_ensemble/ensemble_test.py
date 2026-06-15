"""
Run ensemble experiments for CDC FluSight RSV hospitalization forecasts.
Reference paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11949510/

This script loads predictions from 8 seq2seq models, combines them using several
ensemble methods, evaluates MAE/RMSE against ground truth, and saves prediction
and summary CSV files.

Methods:
    - baseline
    - average ensemble
    - weighted average ensemble
    - linear regression stacking
    - SVR stacking
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # base_pred/2_ensemble/
DATA_DIR   = os.path.dirname(SCRIPT_DIR) # base_pred/
FOLDERS    = list(range(1, 9))
HORIZONS   = [1, 2, 3, 4]
SENTINEL   = -9


# 1. Load and align data
def load_all_models():
    """Load CSVs from folders 1-8 and return a merged wide DataFrame."""
    base_df    = None
    wide_parts = []

    for folder_id in FOLDERS:
        path = os.path.join(DATA_DIR, str(folder_id), f"{folder_id}.csv")
        df = pd.read_csv(path)
        df = df.sort_values(["region", "epiweek"]).reset_index(drop=True)

        if base_df is None:
            base_df = df[["region", "epiweek", "is_test",
                          "true_1", "true_2", "true_3", "true_4"]].copy()

        pred_cols = {f"pred_{h}": f"model_{folder_id}_pred_{h}" for h in HORIZONS}
        part = df[["region", "epiweek"] + [f"pred_{h}" for h in HORIZONS]].rename(
            columns=pred_cols
        )
        wide_parts.append(part)

    merged = base_df.copy()
    for part in wide_parts:
        merged = merged.merge(part, on=["region", "epiweek"], how="left")

    merged = merged.sort_values(["region", "epiweek"]).reset_index(drop=True)
    return merged


# 2. MAE, MSE, and RMSE helpers
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != SENTINEL
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != SENTINEL
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != SENTINEL
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# 3. Average ensemble
def average_ensemble(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    for h in HORIZONS:
        df[f"avg_pred_{h}"] = (
            df.filter(regex=rf"^model_\d+_pred_{h}$")
              .mean(axis=1) # mean for every row horizontally
              .clip(lower=0) # keep it positive
        )
    return df


# 4. Weighted average ensemble (WAE)
def weighted_average_ensemble(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    train_mask = (df["is_test"] == False) # only train datas

    for h in HORIZONS:
        true_col = f"true_{h}" # true_col = groud truth values of this horizion h
        y_train  = df.loc[train_mask, true_col].values

        mse_errors, rmse_errors = [], []
        for m in FOLDERS: # for each horizon, we are chekcing which model are most accurate! (so folder loop inside horizo loop_)
            pred_train = df.loc[train_mask, f"model_{m}_pred_{h}"].values
            mse_errors.append(mse(y_train, pred_train)) 
            # comapre y_train (= ground truth) and pred_train (= model m's predictions)
            rmse_errors.append(rmse(y_train, pred_train))

        mse_errors  = np.array(mse_errors)
        rmse_errors = np.array(rmse_errors)

        # flips errors into weight
        # model with lowest error => gets heightest weight
        w_mse  = 1.0 / (mse_errors  + 1e-8);  w_mse  /= w_mse.sum()
        w_rmse = 1.0 / (rmse_errors + 1e-8);  w_rmse /= w_rmse.sum()

        pred_matrix = df[[f"model_{m}_pred_{h}" for m in FOLDERS]].values
        df[f"wae_mse_pred_{h}"]  = (pred_matrix @ w_mse).clip(min=0)
        df[f"wae_rmse_pred_{h}"] = (pred_matrix @ w_rmse).clip(min=0)

        print(f"  [WAE h={h}] MSE weights:  {dict(zip(FOLDERS, w_mse.round(4)))}")
        print(f"  [WAE h={h}] RMSE weights: {dict(zip(FOLDERS, w_rmse.round(4)))}")

    return df


# 5. Stacking ensemble
def stacking_ensemble(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    for h in HORIZONS:
        true_col     = f"true_{h}"
        feature_cols = [f"model_{m}_pred_{h}" for m in FOLDERS]

        train_idx = df.index[(df["is_test"] == False) & (df[true_col] != SENTINEL)] 
        X_train = df.loc[train_idx, feature_cols].values
        y_train = df.loc[train_idx, true_col].values
        X_all   = df[feature_cols].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        df[f"lr_pred_{h}"] = lr.predict(X_all).clip(min=0)

        svr = SVR()
        svr.fit(X_train, y_train)
        df[f"svr_pred_{h}"] = svr.predict(X_all).clip(min=0)

        print(f"  [Stack h={h}] LR coefs: {lr.coef_.round(4)}  intercept: {lr.intercept_:.4f}")

    return df


# 6. Evaluate all methods
METHOD_LABELS = {
    "baseline": "baseline",
    "avg":      "avg",
    "wae_mse":  "wae_mse",
    "wae_rmse": "wae_rmse",
    "lr":       "lr_stack",
    "svr":      "svr_stack",
}


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split, mask in [("test",  df["is_test"] == True),
                        ("train", df["is_test"] == False)]:
        sub = df[mask]
        for method_key, method_label in METHOD_LABELS.items():
            for h in HORIZONS:
                y_true = sub[f"true_{h}"].values
                y_pred = sub[f"{method_key}_pred_{h}"].values
                records.append({
                    "split":   split,
                    "method":  method_label,
                    "horizon": h,
                    "mae":     mae(y_true, y_pred),
                    "rmse":    rmse(y_true, y_pred),
                })
    return pd.DataFrame(records)


# main
def main():
    print("=" * 60)
    print("Ensemble Experiment — RSV Hospitalization")
    print("=" * 60)

    print("\n[1] Loading model CSVs...")
    merged = load_all_models()
    print(f"    Loaded {len(merged)} rows | "
          f"train={(merged['is_test']==False).sum()}  "
          f"test={(merged['is_test']==True).sum()}")

    print("\n[3] Average ensemble...")
    df = average_ensemble(merged)

    print("\n[4] Weighted average ensemble...")
    df = weighted_average_ensemble(df)

    print("\n[5] Stacking ensemble...")
    df = stacking_ensemble(df)

    # add baseline columns (folder 1 predictions)
    for h in HORIZONS:
        df[f"baseline_pred_{h}"] = df[f"model_1_pred_{h}"]

    print("\n[6] Evaluating...")
    eval_df = evaluate(df)

    # add improvement vs baseline columns
    baseline_ref = (
        eval_df[eval_df["method"] == "baseline"][["split", "horizon", "mae", "rmse"]]
        .rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"})
    )
    eval_df = eval_df.merge(baseline_ref, on=["split", "horizon"], how="left")
    eval_df["mae_vs_baseline"]  = eval_df["mae"]  - eval_df["baseline_mae"]
    eval_df["rmse_vs_baseline"] = eval_df["rmse"] - eval_df["baseline_rmse"]
    eval_df = eval_df.drop(columns=["baseline_mae", "baseline_rmse"])

    test_eval  = eval_df[eval_df["split"] == "test"].drop(columns="split").reset_index(drop=True)
    train_eval = eval_df[eval_df["split"] == "train"].drop(columns="split").reset_index(drop=True)

    print("\n--- TEST SET MAE by method × horizon ---")
    pivot_mae = test_eval.pivot(index="method", columns="horizon", values="mae")
    pivot_mae["avg_horizons"] = pivot_mae.mean(axis=1)
    print(pivot_mae.round(2).to_string())

    print("\n--- TEST SET mae_vs_baseline (negative = improvement) ---")
    pivot_delta = test_eval.pivot(index="method", columns="horizon", values="mae_vs_baseline")
    pivot_delta["avg_horizons"] = pivot_delta.mean(axis=1)
    print(pivot_delta.round(2).to_string())

    print("\n--- TRAIN SET MAE by method × horizon (reference) ---")
    pivot_train = train_eval.pivot(index="method", columns="horizon", values="mae")
    pivot_train["avg_horizons"] = pivot_train.mean(axis=1)
    print(pivot_train.round(2).to_string())

    # Overall averages across horizons
    overall = (
        test_eval
        .groupby("method")[["mae", "rmse", "mae_vs_baseline", "rmse_vs_baseline"]]
        .mean()
        .reset_index()
        .rename(columns={
            "mae":              "avg_mae",
            "rmse":             "avg_rmse",
            "mae_vs_baseline":  "avg_mae_vs_baseline",
            "rmse_vs_baseline": "avg_rmse_vs_baseline",
        })
    )
    method_order = ["baseline", "avg", "wae_mse", "wae_rmse", "lr_stack", "svr_stack"]
    overall = overall.set_index("method").loc[method_order].reset_index()

    # verdict
    baseline_mae  = overall.loc[overall["method"] == "baseline", "avg_mae"].item()
    baseline_rmse = overall.loc[overall["method"] == "baseline", "avg_rmse"].item()

    print("\n--- VERDICT (avg across horizons, test set) ---")
    print(f"  {'method':<12}  {'avg MAE':>9}  {'vs baseline':>12}  {'avg RMSE':>10}  {'vs baseline':>12}")
    print(f"  {'-'*64}")
    for _, row in overall.iterrows():
        mae_tag  = "BETTER" if row["avg_mae_vs_baseline"]  < 0 else ("---" if row["method"] == "baseline" else "WORSE")
        rmse_tag = "BETTER" if row["avg_rmse_vs_baseline"] < 0 else ("---" if row["method"] == "baseline" else "WORSE")
        print(
            f"  {row['method']:<12}  {row['avg_mae']:>9.2f}  "
            f"{row['avg_mae_vs_baseline']:>+10.2f} {mae_tag:<7}  "
            f"{row['avg_rmse']:>10.2f}  "
            f"{row['avg_rmse_vs_baseline']:>+10.2f} {rmse_tag}"
        )

    # save predictions CSV
    pred_cols = (
        ["region", "epiweek", "is_test"]
        + [f"true_{h}"           for h in HORIZONS]
        + [f"model_{m}_pred_{h}" for m in FOLDERS for h in HORIZONS]
        + [f"baseline_pred_{h}"  for h in HORIZONS]
        + [f"avg_pred_{h}"       for h in HORIZONS]
        + [f"wae_mse_pred_{h}"   for h in HORIZONS]
        + [f"wae_rmse_pred_{h}"  for h in HORIZONS]
        + [f"lr_pred_{h}"        for h in HORIZONS]
        + [f"svr_pred_{h}"       for h in HORIZONS]
    )
    pred_out    = os.path.join(SCRIPT_DIR, "ensemble_predictions.csv")
    summary_out = os.path.join(SCRIPT_DIR, "ensemble_summary.csv")
    overall_out = os.path.join(SCRIPT_DIR, "ensemble_overall.csv")

    df[pred_cols].to_csv(pred_out, index=False)
    test_eval.to_csv(summary_out, index=False)
    overall.to_csv(overall_out, index=False)

    print(f"\n[7] Saved predictions -> {pred_out}")
    print(f"    Saved summary     -> {summary_out}")
    print(f"    Saved overall     -> {overall_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
