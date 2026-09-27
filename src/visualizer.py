"""
Module: visualizer
Renders geological well-log composite plots, stratigraphic lithology columns,
and engineering correlation cross-plots between drilling rate and lithological formations.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Professional Geological Color Palette
GEOLOGICAL_COLORS = {
    "SAND": "#f7d560",
    "MEDIUM SAND": "#e5c142",
    "COARSE SAND": "#cba028",
    "CONGLOMERATE": "#d9822b",
    "CLAY": "#7fa655",
    "CLAYSTONE": "#5b8233",
    "GRANODIORITE": "#d95f02",
    "GRANITE": "#e7298a",
    "GRANITE WASH": "#7570b3",
    "MONZONITE": "#386cb0",
    "DACITE": "#6baed6",
    "RHYOLITE": "#9ecae1",
    "DIORITE": "#636363",
    "CEMENT": "#bdbdbd",
    "OBSIDIAN": "#252525",
    "UNKNOWN": "#f0f0f0"
}


def plot_stratigraphic_composite_log(
    lith_df: pd.DataFrame,
    drill_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    depth_limit: int = 1500
):
    """
    Renders an industry-standard dual-track composite well log:
    - Track 1: Drilling Rate / Weight on Bit (continuous curve)
    - Track 2: Lithology Stratigraphic Column (100% stacked area/bar)
    """
    sub_lith = lith_df[lith_df["Depth_To (m)"] <= depth_limit].copy()
    sub_drill = drill_df.iloc[:len(sub_lith)].copy()

    depths = sub_lith["Depth_From (m)"].values
    rock_cols = [c for c in sub_lith.columns if c not in ["Depth_From (m)", "Depth_To (m)"]]

    fig, (ax_drill, ax_lith) = plt.subplots(1, 2, figsize=(14, 10), sharey=True, gridspec_kw={"width_ratios": [1, 2]})

    # Track 1: Drilling Rate Curve
    wob = sub_drill["weight on bit"].values
    ax_drill.plot(wob, depths, color="#08519c", lw=1.5, label="Weight on Bit / ROP")
    ax_drill.set_xlabel("Drilling Rate (Normalized 0-100)", fontsize=11, fontweight="bold")
    ax_drill.set_ylabel("Measured Depth (m)", fontsize=12, fontweight="bold")
    ax_drill.set_xlim(0, 100)
    ax_drill.invert_yaxis()
    ax_drill.grid(True, linestyle=":", alpha=0.6)
    ax_drill.legend(loc="upper right")
    ax_drill.set_title("Track 1: Sensor Log", fontsize=12, fontweight="bold")

    # Track 2: 100% Stacked Lithology Column
    left = np.zeros(len(sub_lith))
    for rock in rock_cols:
        vals = sub_lith[rock].values
        color = GEOLOGICAL_COLORS.get(rock, "#cccccc")
        if vals.sum() > 0:
            ax_lith.barh(depths, vals, left=left, height=5, color=color, label=rock, edgecolor="none")
            left += vals

    ax_lith.set_xlabel("Lithology Composition (%)", fontsize=11, fontweight="bold")
    ax_lith.set_xlim(0, 100)
    ax_lith.grid(True, linestyle=":", alpha=0.6)
    ax_lith.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, frameon=True)
    ax_lith.set_title("Track 2: Subsurface Lithology", fontsize=12, fontweight="bold")

    plt.suptitle(f"Digitized Subsurface Composite Well Log (Interval: 130m - {depth_limit}m)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lithology_composition_breakdown(
    lith_df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Renders an overall formation distribution bar chart across all depth intervals.
    """
    rock_cols = [c for c in lith_df.columns if c not in ["Depth_From (m)", "Depth_To (m)"]]
    sums = lith_df[rock_cols].sum().sort_values(ascending=True)
    total_pct = sums.sum()
    percentages = (sums / total_pct) * 100.0

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [GEOLOGICAL_COLORS.get(rock, "#777777") for rock in percentages.index]
    bars = ax.barh(percentages.index, percentages.values, color=colors, edgecolor="black", alpha=0.85)

    for bar in bars:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Cumulative Stratigraphic Occurrence (%)", fontsize=11, fontweight="bold")
    ax.set_title("Subsurface Lithology Formation Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_drillrate_lithology_correlation(
    lith_df: pd.DataFrame,
    drill_df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Calculates and plots the average drilling rate associated with dominant rock formations.
    """
    rock_cols = [c for c in lith_df.columns if c not in ["Depth_From (m)", "Depth_To (m)", "UNKNOWN"]]
    
    # Identify dominant rock per depth interval
    dominant_rock = lith_df[rock_cols].idxmax(axis=1)
    df_merged = pd.DataFrame({
        "Dominant_Lithology": dominant_rock,
        "Drilling_Rate": drill_df["weight on bit"]
    })

    # Group by lithology and compute statistics
    grouped = df_merged.groupby("Dominant_Lithology")["Drilling_Rate"].agg(["mean", "std", "count"]).dropna()
    grouped = grouped[grouped["count"] >= 5].sort_values(by="mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [GEOLOGICAL_COLORS.get(rock, "#555555") for rock in grouped.index]

    ax.barh(grouped.index, grouped["mean"], xerr=grouped["std"], color=colors, edgecolor="black", alpha=0.85, capsize=4)
    ax.set_xlabel("Mean Drilling Rate / Weight on Bit (with $\\pm 1\\sigma$ STD)", fontsize=11, fontweight="bold")
    ax.set_title("Geoscience Correlation: Drilling Penetration Rate vs. Dominant Lithology", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()
