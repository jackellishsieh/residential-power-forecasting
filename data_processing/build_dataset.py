import numpy as np
import pandas as pd
import pickle
from pathlib import Path

LOAD_DIR = Path(__file__).parent / "load"
META_PATH = Path(__file__).parent / "home_metadata.csv"
OUT_PATH = Path(__file__).parent / "dataset.pkl"
CITIES = ["austin", "california", "newyork"]
CITY_TZ = {
    "austin": "America/Chicago",
    "california": "America/Los_Angeles",
    "newyork": "America/New_York",
}

meta = pd.read_csv(META_PATH).set_index("dataid")["has_car"]

RELABEL_NON_EV = {2335}
DROP = {9278}

dataset = {}
for city in CITIES:
    print(f"Processing {city}...", flush=True)
    df = pd.read_csv(LOAD_DIR / f"15min_load_{city}.csv")
    df["localminute"] = pd.to_datetime(df["localminute"], utc=True)

    for dataid, home_df in df.groupby("dataid"):
        if dataid in DROP:
            continue
        has_car = False if dataid in RELABEL_NON_EV else bool(meta.loc[dataid])
        home_df = (
            home_df.set_index("localminute")[["load", "car1"]]
            .sort_index()
            .dropna(how="all")
        )
        home_df["car1"] = home_df["car1"].round(2)
        if not has_car:
            home_df["car1"] = 0.0
            home_df["charge_state"] = 0.0
        else:
            home_df["charge_state"] = np.where(
                home_df["car1"].isna(), np.nan,
                np.where(home_df["car1"] <= 0.1, 0,
                np.where(home_df["car1"] < 2.0, 1, 2))
            )
        home_df["non_ev_load"] = home_df["load"] - home_df["car1"].fillna(0)

        required = ["load", "car1", "non_ev_load"] if has_car else ["load", "non_ev_load"]
        dates = home_df.index.normalize()
        valid_days = home_df.groupby(dates)[required].apply(lambda g: g.notna().all().all())
        valid_days = valid_days[valid_days].index
        home_df = home_df[home_df.index.normalize().isin(valid_days)]

        local_idx = home_df.index.tz_convert(CITY_TZ[city])
        local_dates = np.array(local_idx.date)
        dst_seconds = np.array([ts.dst().total_seconds() for ts in local_idx])
        dst_df = pd.DataFrame({"date": local_dates, "dst": dst_seconds})
        dst_transition_days = set(
            dst_df.groupby("date")["dst"].nunique().pipe(lambda s: s[s > 1].index)
        )
        home_df = home_df[[d not in dst_transition_days for d in local_dates]]

        dataset[dataid] = (has_car, city, home_df)

with open(OUT_PATH, "wb") as f:
    pickle.dump(dataset, f)

print(f"\nSaved {len(dataset)} homes to {OUT_PATH}")
