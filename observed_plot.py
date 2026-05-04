import os
import pandas as pd
import matplotlib.pyplot as plt

# loading data (pull ground truth)
CSV_PATH = "cdc_datafiles.csv" 
df = pd.read_csv(CSV_PATH) # read in DataFrame formet
print("\nColumns:", list(df.columns))
print(df.head())

# matching the column names with csv files
rename_map = {
    "as of": "as_of",
    "target end date": "target_end_date",
}
df = df.rename(columns=rename_map)

# daes
df["as_of"] = pd.to_datetime(df["as_of"])
df["target_end_date"] = pd.to_datetime(df["target_end_date"])
df["observation"] = pd.to_numeric(df["observation"], errors="coerce") #char to numeric val
df["weekly_rate"] = pd.to_numeric(df["weekly_rate"], errors="coerce")


# target + location
TARGET = "wk inc flu hosp" # weekly incident influenza hospitalizations (num of ppl newly hospitalized during a given week)
LOCATION = "Alabama" # example for now
sub = df[(df["target"] == TARGET) & (df["location_name"] == LOCATION)].copy()
sub = sub.sort_values("target_end_date") 

print("\nNumber of rows after filtering:", len(sub))
if len(sub) == 0:
    raise ValueError("Filtered dataframe is empty. Check target/location names or data availability.")

# plot 1: data overview plot
# observations over target_end_date
plt.figure(figsize=(11, 5))
plt.plot(sub["target_end_date"], sub["observation"], marker="o", label="Observed (count)")
plt.plot(sub["target_end_date"], sub["weekly_rate"], marker="s", linestyle="--", label="Observed (rate)")

plt.xlabel("Week (target_end_date)")
plt.ylabel("Value")
plt.title(f"Overview: Observed flu hospitalizations over time ({LOCATION})")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# plot 2: CDC evalaution sturcture
# as_of is fized
# pick one as_of date ==> show observation by targer_end_date within that snapshot
chosen_as_of = sub["as_of"].max()
snap = sub[sub["as_of"] == chosen_as_of].copy()
snap = snap.sort_values("target_end_date")
print("\nChosen as_of:", chosen_as_of.date())
print("Rows in snapshot:", len(snap))

USE_PREDICTIONS = False
PRED_PATH = "your_submission_predictions.csv" 

if USE_PREDICTIONS:
    pred = pd.read_csv(PRED_PATH)

    # columns like: as_of, target_end_date, location_name, prediction
    pred = pred.rename(columns={
        "as of": "as_of",
        "target end date": "target_end_date",
    })
    pred["as_of"] = pd.to_datetime(pred["as_of"])
    pred["target_end_date"] = pd.to_datetime(pred["target_end_date"])

    # ex. alabama + target only
    pred = pred[(pred["target"] == TARGET) & (pred["location_name"] == LOCATION)].copy()

    # pred falls in 'chosen_as_of'
    pred = pred[pred["as_of"] == chosen_as_of].copy()

    # merge on target_end_date (and location/target already filtered)
    snap = snap.merge(
        pred[["target_end_date", "prediction"]],
        on="target_end_date",
        how="left"
    )

# plot snapshot
plt.figure(figsize=(11, 5))
plt.plot(snap["target_end_date"], snap["observation"], marker="o", label="Observed (count)")

# prediction in one frame
if USE_PREDICTIONS and "prediction" in snap.columns:
    plt.plot(snap["target_end_date"], snap["prediction"], marker="x", label="Submitted prediction")

plt.xlabel("Week (target_end_date)")
plt.ylabel("Weekly incident flu hospitalizations")
plt.title(f"Evaluation-style snapshot (as_of={chosen_as_of.date()}) - {LOCATION}")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
