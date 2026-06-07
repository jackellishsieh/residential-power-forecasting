"""Evaluation: confusion matrices for predicted C and z, plus printing.

Compares predicted (C_hat, z_hat) to ground truth from a labeled dataframe.
Two confusion matrices are produced (specs/model.md §5):

    EV ownership (2×2)   — over homes
    Charging state (3×3) — over (home, day, t) cells, on EV homes only

Aggregation is per-home row-normalized, then mean across homes — so each home
contributes equally regardless of D^(n).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .params import HomeResult, K, STATE_NAMES, T


# ===========================================================================
# Top-level: evaluate() + print_evaluation()
# ===========================================================================

def evaluate(
    df: pd.DataFrame,
    inferences: dict[int, HomeResult],
    c_prob_methods: dict[str, dict[int, float]] | None = None,
    heuristic_states: dict[int, np.ndarray] | None = None,
) -> dict:
    sorted_df = df.sort_values(["home_id", "day", "time_index"])

    ev_hard_cms, ev_soft_cms, ev_heur_cms = [], [], []
    non_ev_hard_cms, non_ev_soft_cms, non_ev_heur_cms = [], [], []
    ev_home_ids, non_ev_home_ids = [], []

    for hid, g in sorted_df.groupby("home_id", sort=True):
        hid = int(hid)
        if hid not in inferences:
            continue
        C_true = int(g["has_ev"].iloc[0])
        D = g["day"].nunique()
        z_true = g["charge_state"].to_numpy().reshape(D, T)
        inf = inferences[hid]

        hard_cm = _per_home_z_confusion_hard(z_true, inf.z_hat)
        soft_cm = (
            _per_home_z_confusion_soft(z_true, inf.z_marginals)
            if inf.z_marginals is not None else None
        )
        heur_cm = None
        if heuristic_states and hid in heuristic_states:
            heur_z  = heuristic_states[hid][: D * T].reshape(D, T)
            heur_cm = _per_home_z_confusion_hard(z_true, heur_z)

        if C_true == 1:
            ev_home_ids.append(hid)
            ev_hard_cms.append(hard_cm)
            if soft_cm is not None:
                ev_soft_cms.append(soft_cm)
            if heur_cm is not None:
                ev_heur_cms.append(heur_cm)
        else:
            non_ev_home_ids.append(hid)
            non_ev_hard_cms.append(hard_cm)
            if soft_cm is not None:
                non_ev_soft_cms.append(soft_cm)
            if heur_cm is not None:
                non_ev_heur_cms.append(heur_cm)

    c_results: dict[str, dict] = {}
    for method_name, c_probs in (c_prob_methods or {}).items():
        c_results[method_name] = _c_confusion_from_probs(sorted_df, inferences, c_probs)

    return {
        "ev_home_ids":   ev_home_ids,
        "ev_z_hard":     _nanmean_cms(ev_hard_cms),
        "ev_z_soft":     _nanmean_cms(ev_soft_cms) if ev_soft_cms else None,
        "ev_z_heur":     _nanmean_cms(ev_heur_cms) if ev_heur_cms else None,
        "non_ev_home_ids": non_ev_home_ids,
        "non_ev_z_hard": _nanmean_cms(non_ev_hard_cms),
        "non_ev_z_soft": _nanmean_cms(non_ev_soft_cms) if non_ev_soft_cms else None,
        "non_ev_z_heur": _nanmean_cms(non_ev_heur_cms) if non_ev_heur_cms else None,
        "c_results":     c_results,
    }


def print_evaluation(results: dict) -> None:
    SEP = "─" * 64

    def _fmt_row(label, row, n=None):
        cells = "  ".join(
            f"{'NaN':>7}" if np.isnan(v) else f"{v:>7.3f}" for v in row
        )
        suffix = f"  (n={n})" if n is not None else ""
        return f"  {label:<8} {cells}{suffix}"

    for group_label, home_ids, hard_cm, soft_cm, heur_cm in [
        ("EV homes (C_true=1)",
         results["ev_home_ids"],
         results["ev_z_hard"], results["ev_z_soft"], results["ev_z_heur"]),
        ("non-EV homes (C_true=0)",
         results["non_ev_home_ids"],
         results["non_ev_z_hard"], results["non_ev_z_soft"], results["non_ev_z_heur"]),
    ]:
        n_homes = len(home_ids)
        print(f"\n{SEP}")
        print(f"z confusion — {group_label}  (N={n_homes} homes)")
        print(f"  Aggregation: per-home row-normalised CM, then mean over {n_homes} homes")
        print(f"  Rows = true state, columns = predicted state")
        if group_label.startswith("non-EV"):
            print("  Note: rows 'low' and 'high' are NaN (no ground-truth examples)")
        header = f"  {'':8}  {'off':>7}  {'low':>7}  {'high':>7}"

        for cm, variant in [(hard_cm, "hard (MAP z)"),
                            (soft_cm, "soft (posterior)"),
                            (heur_cm, "hard (heuristic baseline)")]:
            if cm is None:
                continue
            print(f"\n  [{variant}]")
            print(header)
            for k, name in enumerate(STATE_NAMES):
                print(_fmt_row(name, cm[k]))

    for method_name, cr in results.get("c_results", {}).items():
        print(f"\n{SEP}")
        print(f"C confusion — method: {method_name}")
        print(f"  Aggregation: row-normalised CM averaged over {cr['n_homes']} homes")
        print(f"  ({cr['n_ev']} EV, {cr['n_non_ev']} non-EV)  accuracy={cr['accuracy']:.4f}")
        print(f"  Rows = true C, columns = predicted C")
        header = f"  {'':8}  {'no-EV':>7}  {'EV':>7}"
        for cm, variant in [(cr["hard_cm"], "hard (threshold 0.5)"),
                            (cr["soft_cm"], "soft (P̂ as fraction)")]:
            print(f"\n  [{variant}]")
            print(header)
            for k, name in enumerate(["no-EV", "EV"]):
                print(_fmt_row(name, cm[k]))


def format_confusion(cm, labels):
    """Pretty-print a confusion matrix with row/column totals."""
    n = cm.shape[0]
    header = "          " + "  ".join(f"{l:>8}" for l in labels) + "   total"
    lines = [header]
    for i in range(n):
        row = cm[i]
        lines.append(
            f"  {labels[i]:>6} | " + "  ".join(f"{int(c):>8}" for c in row)
            + f"   {int(row.sum()):>8}"
        )
    col_totals = cm.sum(axis=0)
    lines.append(
        "  " + " " * 6 + " | " + "  ".join(f"{int(c):>8}" for c in col_totals)
        + f"   {int(cm.sum()):>8}"
    )
    return "\n".join(lines)


# ===========================================================================
# Per-home confusion-matrix builders
# ===========================================================================

def _per_home_z_confusion_hard(z_true, z_pred):
    cm = np.full((K, K), np.nan)
    for k_true in range(K):
        mask = (z_true == k_true)
        n = int(mask.sum())
        if n == 0:
            continue
        for k_pred in range(K):
            cm[k_true, k_pred] = float((z_pred[mask] == k_pred).sum()) / n
    return cm


def _per_home_z_confusion_soft(z_true, z_marginals):
    cm = np.full((K, K), np.nan)
    for k_true in range(K):
        mask = (z_true == k_true)
        n = int(mask.sum())
        if n == 0:
            continue
        cm[k_true] = z_marginals[mask].sum(axis=0) / n
    return cm


def _nanmean_cms(cm_list):
    if not cm_list:
        return None
    return np.nanmean(np.stack(cm_list, axis=0), axis=0)


def _c_confusion_from_probs(sorted_df, inferences, c_probs):
    rows = []
    for hid, g in sorted_df.groupby("home_id", sort=True):
        hid = int(hid)
        if hid not in inferences or hid not in c_probs:
            continue
        C_true = int(g["has_ev"].iloc[0])
        p_hat  = float(c_probs[hid])
        rows.append((C_true, int(p_hat >= 0.5), p_hat))

    hard_cm = np.zeros((2, 2), dtype=float)
    soft_cm = np.zeros((2, 2), dtype=float)
    counts  = np.zeros(2, dtype=int)
    for C_true, C_hard, p_hat in rows:
        hard_cm[C_true, C_hard] += 1
        soft_cm[C_true, 0] += 1 - p_hat
        soft_cm[C_true, 1] += p_hat
        counts[C_true] += 1

    with np.errstate(invalid="ignore"):
        hard_cm_norm = np.where(counts[:, None] > 0, hard_cm / counts[:, None], np.nan)
        soft_cm_norm = np.where(counts[:, None] > 0, soft_cm / counts[:, None], np.nan)

    n_correct = sum(1 for C_true, C_hard, _ in rows if C_true == C_hard)
    return {
        "hard_cm":  hard_cm_norm,
        "soft_cm":  soft_cm_norm,
        "accuracy": float(n_correct / max(len(rows), 1)),
        "n_homes":  len(rows),
        "n_ev":     int(counts[1]),
        "n_non_ev": int(counts[0]),
    }
