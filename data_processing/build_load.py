import pandas as pd
from pathlib import Path

IN_DIR = Path(__file__).parent / "resampled"
OUT_DIR = Path(__file__).parent / "load"
OUT_DIR.mkdir(exist_ok=True)
CITIES = ["austin", "california", "newyork"]

for city in CITIES:
    print(f"Processing {city}...", flush=True)
    df = pd.read_csv(IN_DIR / f"15min_data_{city}.csv")

    # load = grid + solar; for homes without solar, solar is NaN so treat as 0
    df["load"] = df["grid"] + df["solar"].fillna(0)
    df = df.drop(columns=["grid", "solar"])

    out_path = OUT_DIR / f"15min_load_{city}.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> {out_path} ({len(df):,} rows)", flush=True)

print("Done.")
