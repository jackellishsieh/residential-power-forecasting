"""Helpers for shuffling the long training dataframe into per-home (D, T) arrays.

Kept private to the package (leading underscore on module name) because the
shape is an internal artefact of the fit/inference pipeline; downstream code
should not depend on it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .params import T


def build_home_arrays(sorted_df: pd.DataFrame, homes: Iterable[int]) -> dict:
    """Reshape long dataframe to per-home (D, T) arrays for total/ev/non-ev/state.

    Required columns: home_id, day, time_index, total_load, ev_load,
                      non_ev_load, charge_state, has_ev, city.

    Returned dict is keyed by home_id and carries:
        has_ev (bool), city, D, x (D,T), x_ev (D,T), x_nev (D,T), z (D,T).
    """
    out = {}
    for hid, g in sorted_df.groupby("home_id", sort=False):
        days = g["day"].to_numpy()
        D = len(np.unique(days))
        if len(g) != D * T:
            raise ValueError(
                f"home {hid}: expected D*T={D*T} rows, got {len(g)} (incomplete days?)"
            )
        out[int(hid)] = {
            "has_ev": bool(g["has_ev"].iloc[0]),
            "city":   g["city"].iloc[0],
            "D":      D,
            "x":      g["total_load"].to_numpy().reshape(D, T).astype(np.float64),
            "x_ev":   g["ev_load"].to_numpy().reshape(D, T).astype(np.float64),
            "x_nev":  g["non_ev_load"].to_numpy().reshape(D, T).astype(np.float64),
            "z":      g["charge_state"].to_numpy().reshape(D, T).astype(np.int64),
        }
    return out
