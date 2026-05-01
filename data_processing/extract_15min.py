import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).parent / "resampled"
CITIES = ["austin", "california", "newyork"]
COLS = ["dataid", "localminute", "car1", "grid", "solar"]

for city in CITIES:
    print(f"Processing {city}...", flush=True)
    path = DATA_DIR / f"1minute_data_{city}" / f"1minute_data_{city}.csv"

    df = pd.read_csv(path, usecols=COLS)
    df["localminute"] = pd.to_datetime(df["localminute"], utc=True).dt.floor("15min")
    df = (
        df.groupby(["dataid", "localminute"])[["car1", "grid", "solar"]]
        .mean()
        .reset_index()
    )

    out_path = OUT_DIR / f"15min_data_{city}.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> {out_path} ({len(df):,} rows)", flush=True)

print("Done.")