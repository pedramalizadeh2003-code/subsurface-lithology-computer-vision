"""
Master CLI Pipeline Runner for Subsurface Lithology & Well-Log Digitization
Loads digitized geological data, validates consistency, runs statistical analysis,
and renders high-resolution well-log composite charts.
"""

import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from src.visualizer import (
    plot_stratigraphic_composite_log,
    plot_lithology_composition_breakdown,
    plot_drillrate_lithology_correlation
)


def main():
    root = Path(__file__).parent
    data_dir = root / "data"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  Subsurface Lithology Computer Vision & Well-Log Digitization Pipeline")
    print("=" * 75)

    lith_path = data_dir / "Lithology.csv"
    drill_path = data_dir / "Drill_Rate.csv"

    if not lith_path.exists() or not drill_path.exists():
        raise FileNotFoundError("Missing Lithology.csv or Drill_Rate.csv in data/ directory.")

    print("\n[+] Loading extracted geological datasets...")
    lith_df = pd.read_csv(lith_path)
    drill_df = pd.read_csv(drill_path)

    n_records = len(lith_df)
    min_depth = lith_df["Depth_From (m)"].min()
    max_depth = lith_df["Depth_To (m)"].max()

    print(f"    - Depth Span: {min_depth}m to {max_depth}m")
    print(f"    - Total Processed Stratigraphic Intervals: {n_records}")
    print(f"    - Extracted Geological Formations: {len(lith_df.columns) - 2} classes")

    # 1. Stratigraphic Composite Log
    print("\n[1/3] Generating Dual-Track Stratigraphic Composite Well Log...")
    out_composite = results_dir / "stratigraphic_lithology_log.png"
    plot_stratigraphic_composite_log(lith_df, drill_df, output_path=out_composite, depth_limit=1500)
    print(f"      [✓] Composite well log exported to: {out_composite.name}")

    # 2. Lithology Composition Overview
    print("\n[2/3] Analyzing Cumulative Formation Distribution...")
    out_comp = results_dir / "lithology_composition_overview.png"
    plot_lithology_composition_breakdown(lith_df, output_path=out_comp)
    print(f"      [✓] Formation distribution chart exported to: {out_comp.name}")

    # 3. Drilling Rate vs. Lithology Correlation
    print("\n[3/3] Computing Geoscience Correlation: Rock Hardness vs. Drilling Rate...")
    out_corr = results_dir / "drillrate_vs_lithology_correlation.png"
    plot_drillrate_lithology_correlation(lith_df, drill_df, output_path=out_corr)
    print(f"      [✓] Correlation analysis exported to: {out_corr.name}")

    print("\n" + "=" * 75)
    print(f"[SUCCESS] All geological charts and analytics successfully exported!")
    print(f"          Visual artifacts stored in: {results_dir.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
