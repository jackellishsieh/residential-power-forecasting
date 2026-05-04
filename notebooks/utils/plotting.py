"""
notebooks/utils/plotting.py

Shared plotting utilities for the residential power forecasting EDA notebooks.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

GRID_COLOR  = "crimson"
SOLAR_COLOR = "mediumseagreen"
LV_COLORS   = {"leg1v": "steelblue", "leg2v": "cornflowerblue"}

# Columns that get fixed colors and are excluded from the appliance palette.
_SPECIAL_COLS = {"dataid", "localminute", "grid", "solar", "leg1v", "leg2v"}


# ── Public: color map builder ──────────────────────────────────────────────────

def build_color_map(df: pd.DataFrame) -> dict[str, tuple]:
    """
    Return a ``{column: color}`` dict for every column in *df* that has at
    least one non-NaN value across all rows.

    Fixed assignments (always applied regardless of whether the column exists):
      • ``grid``  → crimson
      • ``solar`` → mediumseagreen
      • ``leg1v`` → steelblue
      • ``leg2v`` → cornflowerblue

    All remaining active columns are assigned distinct colors drawn from the
    combined ``tab20`` + ``tab20b`` palette (40 colours total), in column order.
    """
    _tab20  = plt.colormaps["tab20"]
    _tab20b = plt.colormaps["tab20b"]
    palette = [_tab20(i) for i in range(20)] + [_tab20b(i) for i in range(20)]

    appliance_cols = [
        c for c in df.columns
        if c not in _SPECIAL_COLS and df[c].notna().any()
    ]

    col_colors: dict[str, tuple] = {
        col: palette[i % len(palette)]
        for i, col in enumerate(appliance_cols)
    }
    col_colors.update({"grid": GRID_COLOR, "solar": SOLAR_COLOR, **LV_COLORS})
    return col_colors


# ── Public: home plotter ───────────────────────────────────────────────────────

def plot_home(
    df: pd.DataFrame,
    col_colors: dict,
    dataid: int,
    timespan: tuple[str, str] | None = None,
    show_voltage: bool = False,
    channels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Plot a structured multi-panel time-series overview for a single home.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset; must contain ``dataid`` and ``localminute`` columns.
    col_colors : dict
        Column → color mapping, e.g. from :func:`build_color_map`.
    dataid : int
        Home identifier to plot.
    timespan : tuple of str, optional
        ``(start, end)`` date strings, e.g. ``('2019-06-01', '2019-08-01')``.
        Interpreted as UTC. If *None* the full available range is shown.
    show_voltage : bool, default False
        When *True*, add a fourth panel showing leg voltages (leg1v / leg2v).
    channels : list of str, optional
        Restrict the appliance panel to these specific channels.
        If *None*, all non-NaN appliance channels for this home are plotted.
    figsize : tuple of (float, float), optional
        Figure size. Defaults to ``(16, 4 × n_panels)``.

    Panel layout
    ------------
    1. **Appliance loads** — every non-grid, non-solar, non-voltage channel
       active for this home, plus a black dotted sum line.
    2. **Power sources** — ``grid`` (red) and ``−solar`` (green, if present),
       plus a black dotted sum representing net grid draw after solar offset.
    3. **Power unaccounted for** — ``grid + solar − Σappliances``; a zero
       reference line is drawn in grey.
    4. **Leg voltages** (optional) — ``leg1v`` and ``leg2v``.
    """
    # ── Prepare slice ──────────────────────────────────────────────────────────
    home = (
        df[df["dataid"] == dataid]
        .sort_values("localminute")
        .set_index("localminute")
    )
    if timespan is not None:
        start = pd.Timestamp(timespan[0]).tz_localize("UTC")
        end   = pd.Timestamp(timespan[1]).tz_localize("UTC")
        home  = home[start:end]

    if home.empty:
        raise ValueError(f"No data for dataid={dataid} in the requested timespan.")

    duration_days = (pd.Timestamp(home.index[-1]) - pd.Timestamp(home.index[0])).total_seconds() / 86_400

    # ── Resolve appliance columns ──────────────────────────────────────────────
    if channels is not None:
        feat_cols = [c for c in channels if c in home.columns and home[c].notna().any()]
    else:
        feat_cols = [
            c for c in home.columns
            if c not in _SPECIAL_COLS and home[c].notna().any()
        ]

    has_grid  = "grid"  in home.columns and home["grid"].notna().any()
    has_solar = "solar" in home.columns and home["solar"].notna().any()

    # ── Layout ─────────────────────────────────────────────────────────────────
    n_panels = 3 + int(show_voltage)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize or (16, 4 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    title = f"dataid {dataid}"
    if timespan is not None:
        title += f"  [{timespan[0]} → {timespan[1]}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_idx = 0

    # ── Panel 1: Appliance loads ───────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1

    feat_sum = pd.Series(0.0, index=home.index)
    for col in feat_cols:
        series = home[col].fillna(0)
        ax.plot(
            home.index, series,
            linewidth=0.4, alpha=0.55,
            color=col_colors.get(col, "gray"),
            label=col,
        )
        feat_sum += series

    # ax.plot(
    #     home.index, feat_sum,
    #     linewidth=1, linestyle="--", color="black", alpha=0.05, label="Σ appliances", zorder=5,
    # )
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Appliance loads ({len(feat_cols)} channels)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _opaque_legend(ax, loc="upper right", fontsize=7, ncol=3)

    # ── Panel 2: Power sources ─────────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1

    source_sum = pd.Series(0.0, index=home.index)
    if has_grid:
        ax.plot(
            home.index, home["grid"],
            linewidth=0.5, alpha=0.7, color=GRID_COLOR, label="grid",
        )
        source_sum += home["grid"].fillna(0)
    if has_solar:
        neg_solar = -home["solar"]
        ax.plot(
            home.index, neg_solar,
            linewidth=0.5, alpha=0.7, color=SOLAR_COLOR, label="−solar",
        )
        source_sum += neg_solar.fillna(0)

    # ax.plot(
    #     home.index, source_sum,
    #     linewidth=1.5, linestyle="--", color="black", alpha=0.5, label="net grid draw", zorder=5,
    # )
    ax.axhline(0, linewidth=0.6, color="gray", alpha=0.6)
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power sources  (grid − solar)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _opaque_legend(ax, loc="upper right", fontsize=8)

    # ── Panel 3: Unaccounted power ─────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1

    grid_vals  = home["grid"].fillna(0)  if has_grid  else pd.Series(0.0, index=home.index)
    solar_vals = home["solar"].fillna(0) if has_solar else pd.Series(0.0, index=home.index)
    unaccounted = grid_vals + solar_vals - feat_sum

    ax.plot(
        home.index, unaccounted,
        linewidth=0.5, color="mediumpurple", label="unaccounted",
    )
    ax.axhline(0, linewidth=0.6, color="gray", alpha=0.6)
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power unaccounted for  (grid + solar − Σ appliances)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _opaque_legend(ax, loc="upper right", fontsize=8)

    # ── Panel 4: Voltage (optional) ────────────────────────────────────────────
    if show_voltage:
        ax = axes[ax_idx]; ax_idx += 1
        lv_present = [c for c in LV_COLORS if c in home.columns and home[c].notna().any()]
        for col in lv_present:
            ax.plot(
                home.index, home[col],
                linewidth=0.5, color=LV_COLORS[col], label=col,
            )
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Leg voltages")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        _opaque_legend(ax, loc="upper right", fontsize=8)

    # ── X-axis ticks ───────────────────────────────────────────────────────────
    _apply_xaxis_ticks(axes[-1], duration_days)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ── Private helpers ────────────────────────────────────────────────────────────

def _opaque_legend(ax: plt.Axes, **legend_kwargs):
    """Create a legend whose handles are always fully opaque and clearly visible."""
    leg = ax.legend(**legend_kwargs)
    # handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    # for h in handles:
    #     h.set_alpha(0.5)
    #     if hasattr(h, "set_linewidth"):
    #         h.set_linewidth(1.5)
    return leg


def _apply_xaxis_ticks(ax: plt.Axes, duration_days: float) -> None:
    """Choose tick density and label format based on the visible time range."""
    if duration_days < 1:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif duration_days < 3:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d  %H:%M"))
    elif duration_days < 14:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    elif duration_days < 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
