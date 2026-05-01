import pandas as pd
from pathlib import Path

RESAMPLED_DIR = Path(__file__).parent / "resampled"
CITIES = ["austin", "california", "newyork"]

records = []
for city in CITIES:
    df = pd.read_csv(RESAMPLED_DIR / f"15min_data_{city}.csv", usecols=["dataid", "car1", "solar"])
    summary = df.groupby("dataid").agg(
        has_car=("car1", lambda s: s.notna().any()),
        has_solar=("solar", lambda s: s.notna().any()),
    ).reset_index()
    summary["city"] = city
    records.append(summary)

meta = pd.concat(records, ignore_index=True)[["dataid", "city", "has_car", "has_solar"]]
out = RESAMPLED_DIR / "home_metadata.csv"
meta.to_csv(out, index=False)

print(meta.groupby("city")[["has_car", "has_solar"]].sum().astype(int))
print(f"\nTotal homes: {len(meta)}")
print(f"Saved to {out}")