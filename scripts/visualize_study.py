#!/usr/bin/env python3
"""Quick visualization script for Bruno abliteration Optuna studies.

Usage:
    python scripts/visualize_study.py                           # Use defaults
    python scripts/visualize_study.py --study qwen32b-v2        # Specific study
    python scripts/visualize_study.py --storage sqlite:///my.db # Custom storage
    python scripts/visualize_study.py --output ./plots          # Custom output dir

This script generates interactive HTML plots from an Optuna study:
- optimization_history.html - Trial progress over time
- pareto_front.html - KL vs Refusals trade-off (multi-objective)
- param_importances.html - Which parameters affect results most
- parallel_coordinate.html - Parameter interactions
- contour.html - 2D parameter relationships
- timeline.html - Trial duration over time
"""

import argparse
import sys
from pathlib import Path

try:
    import optuna
    from optuna.visualization import (
        plot_contour,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_pareto_front,
        plot_slice,
        plot_timeline,
    )
except ImportError:
    print("Error: optuna visualization requires plotly.")
    print("Install with: pip install 'optuna[plotly]'")
    sys.exit(1)


def generate_plots(
    storage: str,
    study_name: str,
    output_dir: Path,
) -> None:
    """Generate all visualization plots from an Optuna study.

    Args:
        storage: Optuna storage URL (e.g., sqlite:///study.db)
        study_name: Name of the study to load
        output_dir: Directory to save HTML plots
    """
    print(f"Loading study '{study_name}' from {storage}...")

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
    except KeyError:
        print(f"Error: Study '{study_name}' not found in {storage}")
        print("\nAvailable studies:")
        summaries = optuna.study.get_all_study_summaries(storage)
        for s in summaries:
            print(f"  - {s.study_name} ({s.n_trials} trials)")
        sys.exit(1)

    # Get completed trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Found {len(completed)} completed trials out of {len(study.trials)} total")

    if len(completed) < 2:
        print("Error: Need at least 2 completed trials for visualization")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {output_dir}/")

    # 1. Optimization History
    print("  Generating optimization_history.html...")
    try:
        fig = plot_optimization_history(study)
        fig.write_html(output_dir / "optimization_history.html")
    except Exception as e:
        print(f"    Warning: Could not generate optimization history: {e}")

    # 2. Pareto Front (for multi-objective)
    if len(study.directions) > 1:
        print("  Generating pareto_front.html...")
        try:
            fig = plot_pareto_front(study)
            fig.write_html(output_dir / "pareto_front.html")
        except Exception as e:
            print(f"    Warning: Could not generate pareto front: {e}")

    # 3. Parameter Importances
    print("  Generating param_importances.html...")
    try:
        fig = plot_param_importances(study)
        fig.write_html(output_dir / "param_importances.html")
    except Exception as e:
        print(f"    Warning: Could not generate param importances: {e}")

    # 4. Parallel Coordinate
    print("  Generating parallel_coordinate.html...")
    try:
        fig = plot_parallel_coordinate(study)
        fig.write_html(output_dir / "parallel_coordinate.html")
    except Exception as e:
        print(f"    Warning: Could not generate parallel coordinate: {e}")

    # 5. Contour plots for key parameter pairs
    print("  Generating contour.html...")
    try:
        # Get parameter names from first trial
        if completed:
            params = list(completed[0].params.keys())
            # Pick two important parameters for contour
            weight_params = [p for p in params if "max_weight" in p]
            if len(weight_params) >= 2:
                fig = plot_contour(study, params=weight_params[:2])
            else:
                fig = plot_contour(study)
            fig.write_html(output_dir / "contour.html")
    except Exception as e:
        print(f"    Warning: Could not generate contour plot: {e}")

    # 6. Slice plots
    print("  Generating slice.html...")
    try:
        fig = plot_slice(study)
        fig.write_html(output_dir / "slice.html")
    except Exception as e:
        print(f"    Warning: Could not generate slice plot: {e}")

    # 7. Timeline
    print("  Generating timeline.html...")
    try:
        fig = plot_timeline(study)
        fig.write_html(output_dir / "timeline.html")
    except Exception as e:
        print(f"    Warning: Could not generate timeline: {e}")

    print("\nDone! Open the HTML files in a browser to view the plots.")
    print(f"\nQuick view: open {output_dir / 'optimization_history.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from a Bruno Optuna study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_study.py
  python scripts/visualize_study.py --study qwen32b-abliteration
  python scripts/visualize_study.py --storage sqlite:///custom.db --output ./my_plots
""",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///heretic_study.db",
        help="Optuna storage URL (default: sqlite:///heretic_study.db)",
    )
    parser.add_argument(
        "--study",
        dest="study_name",
        default="heretic_study",
        help="Study name (default: heretic_study)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./plots"),
        help="Output directory for HTML plots (default: ./plots)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available studies and exit",
    )

    args = parser.parse_args()

    if args.list:
        print(f"Studies in {args.storage}:")
        try:
            summaries = optuna.study.get_all_study_summaries(args.storage)
            for s in summaries:
                print(f"  - {s.study_name}: {s.n_trials} trials")
        except Exception as e:
            print(f"Error accessing storage: {e}")
        return

    generate_plots(
        storage=args.storage,
        study_name=args.study_name,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
