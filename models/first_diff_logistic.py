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
    load: np.ndarray, window: int, low_thresh: float, high_thresh: float,
    max_duration: int = 48,
) -> np.ndarray:
    """Smooth load with rolling median then run delta-based state machine -> states in {0,1,2}.

    max_duration: max consecutive steps allowed in a non-zero state before forcing back to 0.
    """
    smoothed = pd.Series(load).rolling(window, min_periods=1, center=True).median().to_numpy()
    delta = np.diff(smoothed, prepend=smoothed[0])
    state, states = 0, np.empty(len(load), dtype=int)
    steps_in_state = 0
    for t in range(len(load)):
        d = delta[t]
        if state == 0:
            if d >= high_thresh:
                state = 2
                steps_in_state = 0
            elif d >= low_thresh:
                state = 1
                steps_in_state = 0
        elif state == 1:
            steps_in_state += 1
            if steps_in_state >= max_duration:
                state = 0
                steps_in_state = 0
            elif d >= high_thresh - low_thresh:
                state = 2
                steps_in_state = 0
            elif d <= -low_thresh:
                state = 0
                steps_in_state = 0
        else:  # state == 2
            steps_in_state += 1
            if steps_in_state >= max_duration:
                state = 0
                steps_in_state = 0
            elif d <= -high_thresh:
                state = 0
                steps_in_state = 0
            elif d <= -low_thresh:
                state = 1
                steps_in_state = 0
        states[t] = state
    return states


def transitions_per_day(
    load: np.ndarray, window: int, low_thresh: float, high_thresh: float,
    max_duration: int = 48,
) -> float:
    states = estimate_ev_state(load, window, low_thresh, high_thresh, max_duration)
    return (np.diff(states) != 0).sum() / (len(load) / 96)


def tune(train_homes: dict) -> tuple[LogisticRegression, int, float, float, int]:
    """Grid search (window, low, high, max_duration) on training homes; returns fitted model and best params."""
    loads = [df["load"].to_numpy() for _, (_, _, df) in train_homes.items()]
    y = np.array([int(has_car) for _, (has_car, _, _) in train_homes.items()])

    best_score, best_model, best = -1.0, None, None
    for w in [2, 4, 6]:
        for lo in np.arange(0.4, 1.6, 0.4):
            for hi in np.arange(1.2, 3.2, 0.4):
                if hi <= lo:
                    continue
                for md in [16, 32, 48]:  # 4h, 8h, 12h
                    X = [[transitions_per_day(load, w, lo, hi, md)] for load in loads]
                    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
                    score = average_precision_score(y, model.predict_proba(X)[:, 1])
                    if score > best_score:
                        best_score, best_model, best = score, model, (w, lo, hi, md)

    w, lo, hi, md = best
    print(f"Best: window={w} ({w*15} min), low={lo:.1f}, high={hi:.1f}, max_duration={md} ({md*15} min)  (train avg precision={best_score:.4f})")
    return best_model, w, lo, hi, md


def predict(
    model: LogisticRegression,
    test_homes: dict,
    window: int,
    low_thresh: float,
    high_thresh: float,
    max_duration: int = 48,
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
        states = estimate_ev_state(load, window, low_thresh, high_thresh, max_duration)
        rate = (np.diff(states) != 0).sum() / (len(load) / 96)
        p_hat = model.predict_proba([[rate]])[0, 1]
        rows.append({"dataid": dataid, "has_ev": int(has_car), "transitions_per_day": round(rate, 3), "p_hat": round(p_hat, 3)})
        charge_states[dataid] = states
    return pd.DataFrame(rows), charge_states


def save(model: LogisticRegression, window: int, low_thresh: float, high_thresh: float, max_duration: int, path) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump({"model": model, "window": window, "low_thresh": low_thresh, "high_thresh": high_thresh, "max_duration": max_duration}, f)


def load(path) -> tuple[LogisticRegression, int, float, float, int]:
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["window"], d["low_thresh"], d["high_thresh"], d["max_duration"]
