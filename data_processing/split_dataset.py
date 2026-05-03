"""
Creates a fixed train/test split and saves two pkl files.

  Train: 9 EV + 45 non-EV = 54 homes  -> splits/train.pkl
  Test:  3 EV + 15 non-EV = 18 homes  -> splits/test.pkl
"""

import pickle
import numpy as np
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.pkl"
SPLITS_DIR   = Path(__file__).parent / "splits"
SEED = 0

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)

ev_ids   = [d for d, (has_car, *_) in dataset.items() if has_car]
noev_ids = [d for d, (has_car, *_) in dataset.items() if not has_car]

rng = np.random.default_rng(SEED)
rng.shuffle(ev_ids)
rng.shuffle(noev_ids)

train = {d: dataset[d] for d in ev_ids[:9]  + noev_ids[:45]}
test  = {d: dataset[d] for d in ev_ids[9:]  + noev_ids[45:]}

SPLITS_DIR.mkdir(exist_ok=True)
with open(SPLITS_DIR / "train.pkl", "wb") as f:
    pickle.dump(train, f)
with open(SPLITS_DIR / "test.pkl", "wb") as f:
    pickle.dump(test, f)

print(f"Train: {len(train)} homes ({sum(v[0] for v in train.values())} EV)")
print(f"Test:  {len(test)} homes ({sum(v[0] for v in test.values())} EV)")
print(f"Saved to {SPLITS_DIR}")
