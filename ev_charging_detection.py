"""
EV Charging Detection via Adaptive Baseline and Threshold-Based Interval Detection
===================================================================================

This script implements the methodology described in:

    "EV Charging Detection from Solar Inverter Data using an Adaptive Baseline
     and Threshold-Based Interval Detection"

Overview
--------
Given a 15-minute resolution total electricity consumption time series recorded
by a residential SMA solar inverter, the algorithm:

    1. Estimates an *adaptive baseline* that tracks the building's normal
       (non-EV-charging) consumption level in a month-aware, rolling fashion.
    2. Derives a *charging threshold* = baseline + fixed offset (2 000 W).
    3. Detects *EV charging intervals* as contiguous runs of time steps where
       consumption exceeds the threshold, subject to minimum-length and
       maximum-consecutive-false constraints.

Dependencies
------------
    pandas >= 1.5
    numpy  >= 1.23
    matplotlib >= 3.6

Usage
-----
    python ev_charging_detection.py

Output figures are saved to OUTPUT_DIR (see Configuration section below).
"""

# =============================================================================
# SECTION 0 — Imports
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# =============================================================================
# SECTION 1 — Configuration
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE  = os.path.join(_SCRIPT_DIR, "inversor_data.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "figures")

# ── Adaptive Baseline Parameters ─────────────────────────────────────────────
BASELINE_SIGNAL_COL          = "Consumption"          # column to baseline
BASELINE_OUTPUT_COL          = "Baseline_Adaptive"    # output column name
BASELINE_METHOD              = "mean"                 # "mean" | "median" | "percentile"
BASELINE_DAYS_WINDOW         = 4                      # rolling window width in days
BASELINE_N_POINTS_CHECK      = 3                      # isolation check half-window

# ── Detection Parameters ──────────────────────────────────────────────────────
DETECTION_MIN_LENGTH         = 10     # minimum number of consecutive time steps
DETECTION_MAX_CONSEC_FALSE   = 2      # max consecutive steps below threshold inside interval
DETECTION_THRESHOLD_MULT     = 0.9    # relaxed threshold multiplier for interior points
DETECTION_THRESHOLD_OFFSET   = 300    # alternative relaxed threshold: threshold - offset (W)

# ── Figure Date Ranges ────────────────────────────────────────────────────────
FIG1_START = "2024-08-29 00:00:00"   # range for baseline/threshold overview plot
FIG1_END   = "2024-09-01 00:00:00"

FIG2_START = "2024-08-29 00:00:00"   # range for detected-intervals plot
FIG2_END   = "2024-09-01 00:00:00"

# =============================================================================
# SECTION 2 — Adaptive Baseline Algorithm
# =============================================================================

def _min_interpolation(series: pd.Series) -> pd.Series:
    """
    Conservative gap-filling: at each NaN position the filled value is the
    *minimum* of the forward-filled and backward-filled values, preventing the
    baseline from overshooting during detected charging events.
    """
    fwd = series.ffill()
    bwd = series.bfill()
    return np.minimum(fwd, bwd)


def compute_adaptive_baseline(
    df: pd.DataFrame,
    signal_col: str = "Consumption",
    new_col: str = "Baseline_Adaptive",
    normal_baseline_method: str = "mean",
    days_normal_baseline: int = 4,
    n_points_check: int = 3,
) -> pd.DataFrame:
    """
    Compute a month-aware, adaptive baseline for a consumption time series.

    Algorithm Overview
    ------------------
    1. **Month configuration** — Each calendar month has two tuning knobs:
       - *percentile*: used to compute a coarse month-level baseline as a
         fallback fill value.
       - *threshold_multiplier*: scales the rolling normal baseline into a
         threshold that separates normal from abnormal (EV-charging) periods.

    2. **Rolling normal baseline** — A centered rolling window of width
       ``days_normal_baseline`` days (assuming 15-min sampling, i.e. 4 pts/h)
       computes the mean (or median/percentile) of the raw signal, producing
       the ``normal_baseline`` reference series.

    3. **Normal-period mask** — A sample is "normal" if its value is at or
       below ``normal_baseline``.  An additional isolation check discards
       isolated True flags: if a point is True but fewer than 2 of its
       ±``n_points_check`` neighbours are also True, it is reclassified as
       not-normal, preventing the baseline from being pulled up by brief
       consumption spikes.

    4. **Short rolling median on normal periods** — For samples classified as
       normal, a short 7-point centered rolling median is computed.  Non-normal
       samples are set to NaN.

    5. **Gap-filling** — NaN gaps (EV charging events) are filled using
       ``_min_interpolation`` (min of forward/backward fill), then any
       remaining NaN (at series edges) is filled with the coarse month baseline.

    6. **Charging threshold** — ``Charging_Threshold = new_col + 2000`` (W) is
       appended as a ready-to-use detection threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (15-min resolution assumed).
    signal_col : str
        Name of the consumption column.
    new_col : str
        Name of the output adaptive baseline column.
    normal_baseline_method : {"mean", "median", "percentile"}
        Statistic used for the rolling normal baseline reference.
    days_normal_baseline : int
        Rolling window width expressed in days (converted to samples internally).
    n_points_check : int
        Half-width of the isolation check window (samples).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional columns:
        ``normal_baseline``, ``normal_threshold``, ``not_normal``,
        ``new_col``, ``Charging_Threshold``.
    """
    df = df.copy()
    df["month"] = df.index.month

    # ── 1. Month-adaptive configuration ───────────────────────────────────────
    month_config = {
        1:  {"percentile": 0.30, "threshold_multiplier": 1.8},
        2:  {"percentile": 0.30, "threshold_multiplier": 1.8},
        3:  {"percentile": 0.15, "threshold_multiplier": 2.0},
        4:  {"percentile": 0.20, "threshold_multiplier": 1.8},
        5:  {"percentile": 0.20, "threshold_multiplier": 1.8},
        6:  {"percentile": 0.15, "threshold_multiplier": 1.8},
        7:  {"percentile": 0.15, "threshold_multiplier": 1.8},
        8:  {"percentile": 0.15, "threshold_multiplier": 1.8},
        9:  {"percentile": 0.20, "threshold_multiplier": 1.8},
        10: {"percentile": 0.20, "threshold_multiplier": 1.8},
        11: {"percentile": 0.10, "threshold_multiplier": 5.0},
        12: {"percentile": 0.25, "threshold_multiplier": 1.8},
    }

    # ── 2. Coarse month baselines (fallback fill values) ──────────────────────
    month_baselines = {}
    for m in range(1, 13):
        data_m = df[df["month"] == m]
        if len(data_m) > 0:
            month_baselines[m] = data_m[signal_col].quantile(month_config[m]["percentile"])
    df["month_baseline"] = df["month"].map(month_baselines)

    # ── 3. Rolling normal baseline ────────────────────────────────────────────
    window = days_normal_baseline * 24 * 4  # samples (15-min data)
    rolling_obj = df[signal_col].rolling(window=window, center=True, min_periods=1)

    if normal_baseline_method == "percentile":
        pct_map = {m: month_config[m]["percentile"] for m in range(1, 13)}
        df["normal_baseline"] = [
            r.quantile(p) if len(r) > 0 else np.nan
            for r, p in zip(rolling_obj, df["month"].map(pct_map))
        ]
    elif normal_baseline_method == "median":
        df["normal_baseline"] = rolling_obj.median()
    else:  # default: mean
        df["normal_baseline"] = rolling_obj.mean()

    # ── 4. Normal-period mask (with isolation check) ──────────────────────────
    df["normal_threshold"] = df["normal_baseline"]
    normal_mask = df[signal_col] <= df["normal_threshold"]

    # Isolation check: discard isolated True flags
    window_size = n_points_check * 2 + 1
    expanded = (
        normal_mask.rolling(window=window_size, center=True, min_periods=1)
        .sum()
        .astype(int) > 1
    )
    normal_mask = normal_mask & expanded

    # ── 5. Short rolling median on normal periods ─────────────────────────────
    normal_consumption = df[signal_col].where(normal_mask)
    rolling_baseline = normal_consumption.rolling(window=7, min_periods=1, center=True).median()

    df[new_col] = np.where(normal_mask, rolling_baseline, np.nan)
    df["not_normal"] = ~normal_mask

    # ── 6. Gap-filling ────────────────────────────────────────────────────────
    df[new_col] = _min_interpolation(df[new_col])
    df[new_col] = df[new_col].fillna(df["month_baseline"])

    # ── 7. Charging threshold ─────────────────────────────────────────────────
    df["Charging_Threshold"] = df[new_col] + 2000

    return df.drop(columns=["month"], errors="ignore")


# =============================================================================
# SECTION 3 — Interval Detection Algorithm
# =============================================================================

def detect_charging_intervals(
    ts_value: pd.Series,
    threshold: pd.Series,
    min_length: int = 10,
    max_consec_false: int = 2,
    threshold_multiplier: float = 0.9,
    threshold_offset: float = 300,
) -> list:
    """
    Detect EV charging intervals as contiguous runs where consumption exceeds a
    threshold.

    Detection Logic
    ---------------
    The detector scans the time series left-to-right with the following rules:

    *Interval start:* Two consecutive samples must both satisfy
    ``ts_value >= threshold``.

    *Minimum length:* The candidate window of ``min_length`` steps is
    evaluated:

    - First two steps: strict condition ``ts_value >= threshold``.
    - Interior steps: relaxed condition
      ``ts_value >= max(threshold × multiplier, threshold − offset)``.

    A candidate window is accepted if:
    (a) The ratio of failing steps does not exceed 30 %, AND
    (b) No run of consecutive failing steps exceeds ``max_consec_false``, AND
    (c) The last step satisfies the strict condition.

    *Expansion:* Accepted intervals are extended right as long as the relaxed
    condition holds (subject to the same false-ratio and consecutive-false
    constraints).

    *Left expansion:* The interval is also extended left as far as the strict
    condition holds.

    Parameters
    ----------
    ts_value : pd.Series
        Time series of consumption values.
    threshold : pd.Series or float
        Detection threshold (per-sample or scalar).
    min_length : int
        Minimum number of time steps for a valid interval.
    max_consec_false : int
        Maximum number of consecutive below-threshold steps allowed inside a
        detected interval.
    threshold_multiplier : float
        Relaxed threshold = ``threshold × threshold_multiplier``.
    threshold_offset : float
        Alternative relaxed threshold = ``threshold − threshold_offset`` (W).

    Returns
    -------
    list of bool
        Per-sample boolean flags; ``True`` indicates the sample belongs to a
        detected EV charging interval.
    """
    n = len(ts_value)
    flags = [False] * n
    i = 0

    if not isinstance(threshold, pd.Series):
        threshold = pd.Series([threshold] * n, index=ts_value.index)

    false_ratio = 0.3
    acc = 0

    while i < n - 1:
        if acc == 0:
            # Two-point strict entry condition
            if not (
                ts_value.iloc[i] >= threshold.iloc[i]
                and ts_value.iloc[i + 1] >= threshold.iloc[i + 1]
            ):
                i += 1
                continue
            start = i
            end = i + min_length
            if end > n:
                break

        acc = 0

        # Evaluate minimum-length window
        w_val = ts_value.iloc[start:end]
        w_thr = threshold.iloc[start:end]

        condition = pd.Series([False] * len(w_val), index=w_val.index)
        condition.iloc[0] = w_val.iloc[0] >= w_thr.iloc[0]
        condition.iloc[1] = w_val.iloc[1] >= w_thr.iloc[1]

        relaxed_thr = pd.Series.combine(
            w_thr.iloc[2:] * threshold_multiplier,
            w_thr.iloc[2:] - threshold_offset,
            max,
        )
        condition.iloc[2:] = w_val.iloc[2:] >= relaxed_thr

        false_count = int((~condition).sum())
        max_false   = int(false_ratio * len(condition))
        consec_false = 0
        valid = False

        for val in condition:
            consec_false = 0 if val else consec_false + 1
            if consec_false > max_consec_false:
                break
        else:
            if false_count <= max_false and w_val.iloc[-1] >= w_thr.iloc[-1]:
                valid = True

        if valid:
            # Expand rightward
            j = end
            consec_false = 0
            while j < n:
                ok = ts_value.iloc[j] >= threshold.iloc[j] * threshold_multiplier
                consec_false = 0 if ok else consec_false + 1
                if not ok:
                    false_count += 1
                interval_len = j - start + 1
                if false_count > int(false_ratio * interval_len) or consec_false > max_consec_false:
                    break
                j += 1

            # Trim so last point satisfies strict condition
            while j > start and not (ts_value.iloc[j - 1] >= threshold.iloc[j - 1]):
                j -= 1

            for k in range(start, j):
                flags[k] = True

            # Expand leftward (strict condition)
            left = start - 1
            while left >= 0 and ts_value.iloc[left] >= threshold.iloc[left]:
                flags[left] = True
                left -= 1

            i = j
        else:
            # Two-point lookahead
            extra = False
            for offset in range(1, 3):
                if (
                    end + offset - 1 < n
                    and ts_value.iloc[end + offset - 1] >= threshold.iloc[end + offset - 1]
                ):
                    extra = True
                    break
            if extra:
                acc = 1
                end = end + offset
            else:
                i += 1

    return flags


# =============================================================================
# SECTION 4 — Plotting Functions
# =============================================================================

def plot_baseline_and_threshold(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    consumption_col: str = "Consumption",
    output_path: str = None,
    figsize: tuple = (16, 6),
) -> None:
    """
    Plot consumption, normal baseline, adaptive baseline and charging threshold
    over a specified date range.

    Columns used (if present in df):
        - ``consumption_col``    → black solid line
        - ``normal_baseline``    → orange dashed
        - ``Baseline_Adaptive``  → cyan dashed
        - ``Charging_Threshold`` → pink dotted

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`compute_adaptive_baseline`.
    start_date, end_date : str
        ISO-format datetime strings defining the plot window.
    consumption_col : str
        Name of the consumption column.
    output_path : str, optional
        File path to save the figure (SVG).  If None the figure is only shown.
    figsize : tuple
        Matplotlib figure size.
    """
    df_plot = df.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df_plot.index, df_plot[consumption_col],
            color="black", linewidth=1.5, label="Consumption")

    if "normal_baseline" in df_plot.columns:
        ax.plot(df_plot.index, df_plot["normal_baseline"],
                color="#FF4400", linestyle="--", linewidth=2.0,
                label="Normal Baseline", alpha=0.95)

    if "Baseline_Adaptive" in df_plot.columns:
        ax.plot(df_plot.index, df_plot["Baseline_Adaptive"],
                color="#00C3FF", linestyle="--", linewidth=2.0,
                label="Adaptive Baseline", alpha=0.95)

    if "Charging_Threshold" in df_plot.columns:
        ax.plot(df_plot.index, df_plot["Charging_Threshold"],
                color="#FF009D", linestyle=":", linewidth=2.0,
                label="Charging Threshold", alpha=0.95)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_xlabel("Date and Time", fontsize=18, fontweight="bold")
    ax.set_ylabel("Power (W)", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Consumption, Baselines and Threshold\n({start_date[:10]} — {end_date[:10]})",
        fontsize=18, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close()


def plot_detected_intervals(
    df: pd.DataFrame,
    flags_col: str,
    start_date: str,
    end_date: str,
    consumption_col: str = "Consumption",
    output_path: str = None,
    figsize: tuple = (16, 6),
) -> None:
    """
    Plot the consumption time series with detected EV charging intervals
    highlighted in red over a specified date range.

    The base consumption trace is drawn in black.  Each contiguous run of True
    flags is re-drawn in red on top.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``consumption_col`` and ``flags_col``.
    flags_col : str
        Boolean column produced by :func:`detect_charging_intervals`.
    start_date, end_date : str
        ISO-format datetime strings defining the plot window.
    consumption_col : str
        Name of the consumption column.
    output_path : str, optional
        File path to save the figure (SVG).  If None the figure is only shown.
    figsize : tuple
        Matplotlib figure size.
    """
    df_plot = df.loc[start_date:end_date].copy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df_plot.index, df_plot[consumption_col],
            color="black", linewidth=1.5, label="Consumption", zorder=2)

    if flags_col in df_plot.columns:
        flag_values   = df_plot[flags_col].values
        cons_values   = df_plot[consumption_col].values
        times         = df_plot.index.values

        in_interval       = False
        interval_start    = None

        for idx in range(len(flag_values)):
            if flag_values[idx] and not in_interval:
                interval_start = idx
                in_interval    = True
            elif not flag_values[idx] and in_interval:
                ax.plot(
                    times[interval_start:idx],
                    cons_values[interval_start:idx],
                    color="#FF0000", linewidth=2.5, alpha=0.9, zorder=3,
                )
                in_interval = False

        if in_interval:
            ax.plot(
                times[interval_start:],
                cons_values[interval_start:],
                color="#FF0000", linewidth=2.5, alpha=0.9, zorder=3,
            )

        red_patch = Patch(color="#FF0000", label="Detected EV Charging Intervals", alpha=0.9)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(red_patch)
        ax.legend(handles=handles, loc="best", fontsize=14, framealpha=0.95)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_xlabel("Date and Time", fontsize=18, fontweight="bold")
    ax.set_ylabel("Power (W)", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Consumption with Detected EV Charging Intervals\n({start_date[:10]} — {end_date[:10]})",
        fontsize=18, fontweight="bold", pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close()


# =============================================================================
# SECTION 5 — Main Pipeline
# =============================================================================

if __name__ == "__main__":

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["Datetime"], index_col="Datetime")
    df.rename(columns={"TotalConsumption(W)": BASELINE_SIGNAL_COL}, inplace=True)
    print(f"[INFO] Loaded {len(df):,} rows  ({df.index[0]} -> {df.index[-1]})")

    # ── 2. Compute adaptive baseline + charging threshold ─────────────────────
    print("[INFO] Computing adaptive baseline …")
    df = compute_adaptive_baseline(
        df,
        signal_col=BASELINE_SIGNAL_COL,
        new_col=BASELINE_OUTPUT_COL,
        normal_baseline_method=BASELINE_METHOD,
        days_normal_baseline=BASELINE_DAYS_WINDOW,
        n_points_check=BASELINE_N_POINTS_CHECK,
    )
    print(f"[INFO] Baseline computed. Charging threshold = Adaptive Baseline + 2 000 W")

    # ── 3. Detect EV charging intervals ──────────────────────────────────────
    print("[INFO] Detecting EV charging intervals …")
    flags = detect_charging_intervals(
        ts_value=df[BASELINE_SIGNAL_COL],
        threshold=df["Charging_Threshold"],
        min_length=DETECTION_MIN_LENGTH,
        max_consec_false=DETECTION_MAX_CONSEC_FALSE,
        threshold_multiplier=DETECTION_THRESHOLD_MULT,
        threshold_offset=DETECTION_THRESHOLD_OFFSET,
    )
    df["flags"] = flags
    n_detected = sum(flags)
    print(f"[INFO] Detected {n_detected:,} flagged time steps "
          f"({n_detected * 15 / 60:.1f} h of EV charging detected)")

    # ── 4. Figure 1 — Baseline & threshold overview ───────────────────────────
    print("[INFO] Generating Figure 1: Baseline and Threshold …")
    plot_baseline_and_threshold(
        df,
        start_date=FIG1_START,
        end_date=FIG1_END,
        consumption_col=BASELINE_SIGNAL_COL,
        output_path=os.path.join(OUTPUT_DIR, "fig1_baseline_and_threshold.svg"),
    )

    # ── 5. Figure 2 — Detected intervals ─────────────────────────────────────
    print("[INFO] Generating Figure 2: Detected EV Charging Intervals …")
    plot_detected_intervals(
        df,
        flags_col="flags",
        start_date=FIG2_START,
        end_date=FIG2_END,
        consumption_col=BASELINE_SIGNAL_COL,
        output_path=os.path.join(OUTPUT_DIR, "fig2_detected_intervals.svg"),
    )

    print("[INFO] Done.")
