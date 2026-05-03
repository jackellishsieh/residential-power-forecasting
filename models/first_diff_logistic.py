"""
Probabilistic forecast of EV ownership from a home's load signal.

1. Smooth load with a rolling median, take first diff.
2. Run state machine {0,1,2} on smoothed diff to estimate charge state.
3. Count state transitions per day as a home-level feature.
4. Logistic regression: transitions/day -> P(has_EV).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def estimate_ev_state(
    load: np.ndarray, window: int, low_thresh: float, high_thresh: float
) -> np.ndarray:
    """Smooth load with rolling median then run delta-based state machine -> states in {0,1,2}."""
    smoothed = pd.Series(load).rolling(window, min_periods=1).median().to_numpy()
    delta = np.diff(smoothed, prepend=smoothed[0])
    state, states = 0, np.empty(len(load), dtype=int)
    for t in range(len(load)):
        d = delta[t]
        if state == 0:
            if d >= high_thresh:
                state = 2
            elif d >= low_thresh:
                state = 1
        elif state == 1:
            if d >= high_thresh - low_thresh:
                state = 2
            elif d <= -low_thresh:
                state = 0
        else:
            if d <= -high_thresh:
                state = 0
            elif d <= -low_thresh:
                state = 1
        states[t] = state
    return states


def transitions_per_day(
    load: np.ndarray, window: int, low_thresh: float, high_thresh: float
) -> float:
    states = estimate_ev_state(load, window, low_thresh, high_thresh)
    return (np.diff(states) != 0).sum() / (len(load) / 96)


def tune(train_homes: dict) -> tuple[LogisticRegression, int, float, float]:
    """Grid search (window, low, high) on training homes; returns fitted model and best params."""
    loads = [df["load"].to_numpy() for _, (_, _, df) in train_homes.items()]
    y = np.array([int(has_car) for _, (has_car, _, _) in train_homes.items()])

    best_auroc, best = -1.0, None
    for w in [2, 4, 8, 16, 24, 48]:
        for lo in np.arange(0.4, 1.6, 0.2):
            for hi in np.arange(1.2, 3.2, 0.2):
                if hi <= lo:
                    continue
                X = [[transitions_per_day(load, w, lo, hi)] for load in loads]
                model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
                    X, y
                )
                score = average_precision_score(y, model.predict_proba(X)[:, 1])
                if score > best_auroc:
                    best_auroc, best = score, (w, lo, hi)

    w, lo, hi = best
    X = [[transitions_per_day(load, w, lo, hi)] for load in loads]
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
    print(
        f"Best: window={w} ({w*15} min), low={lo:.1f}, high={hi:.1f}  (train AUROC={best_auroc:.4f})"
    )
    return model, w, lo, hi


def predict(
    model: LogisticRegression,
    test_homes: dict,
    window: int,
    low_thresh: float,
    high_thresh: float,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Returns
    -------
    summary      : DataFrame with one row per home (dataid, has_ev, transitions_per_day, p_hat)
    charge_states: {dataid: per-timestep state array in {0, 1, 2}}
    """
    rows, charge_states = [], {}
    for dataid, (has_car, _, df) in test_homes.items():
        load = df["load"].to_numpy()
        states = estimate_ev_state(load, window, low_thresh, high_thresh)
        rate = (np.diff(states) != 0).sum() / (len(load) / 96)
        p_hat = model.predict_proba([[rate]])[0, 1]
        rows.append(
            {
                "dataid": dataid,
                "has_ev": int(has_car),
                "transitions_per_day": round(rate, 3),
                "p_hat": round(p_hat, 3),
            }
        )
        charge_states[dataid] = states
    return pd.DataFrame(rows), charge_states


def save(model: LogisticRegression, window: int, low_thresh: float, high_thresh: float, path) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump({"model": model, "window": window, "low_thresh": low_thresh, "high_thresh": high_thresh}, f)


def load(path) -> tuple[LogisticRegression, int, float, float]:
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["window"], d["low_thresh"], d["high_thresh"]
