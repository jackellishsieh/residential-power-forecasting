"""
Creates a fixed stratified train/test split (~75/25) and saves two pkl files.
EV and non-EV homes are split independently so both sets have the same EV ratio.
"""

import pickle
import numpy as np
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.pkl"
SPLITS_DIR   = Path(__file__).parent / "splits"
SEED = 0
TRAIN_FRAC = 0.75

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)

ev_ids   = [d for d, (has_car, *_) in dataset.items() if has_car]
noev_ids = [d for d, (has_car, *_) in dataset.items() if not has_car]

rng = np.random.default_rng(SEED)
rng.shuffle(ev_ids)
rng.shuffle(noev_ids)

n_ev_train   = round(len(ev_ids)   * TRAIN_FRAC)
n_noev_train = round(len(noev_ids) * TRAIN_FRAC)

train = {d: dataset[d] for d in ev_ids[:n_ev_train]   + noev_ids[:n_noev_train]}
test  = {d: dataset[d] for d in ev_ids[n_ev_train:]   + noev_ids[n_noev_train:]}

SPLITS_DIR.mkdir(exist_ok=True)
with open(SPLITS_DIR / "train.pkl", "wb") as f:
    pickle.dump(train, f)
with open(SPLITS_DIR / "test.pkl", "wb") as f:
    pickle.dump(test, f)

print(f"Train: {len(train)} homes ({sum(v[0] for v in train.values())} EV)")
print(f"Test:  {len(test)} homes ({sum(v[0] for v in test.values())} EV)")
print(f"Saved to {SPLITS_DIR}")
