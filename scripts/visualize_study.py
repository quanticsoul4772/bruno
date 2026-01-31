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

    # Check if multi-objective
    is_multi_objective = len(study.directions) > 1
    if is_multi_objective:
        print(f"Multi-objective study detected ({len(study.directions)} objectives)")
        print("  Objective 0: KL Divergence (minimize)")
        print("  Objective 1: Refusals (minimize)")
        # Target functions for each objective
        target_kl = lambda t: t.values[0] if t.values else float("inf")
        target_refusals = (
            lambda t: t.values[1] if t.values and len(t.values) > 1 else float("inf")
        )
        objectives = [
            ("kl", "KL Divergence", target_kl),
            ("refusals", "Refusals", target_refusals),
        ]
    else:
        objectives = [("value", "Objective", None)]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {output_dir}/")

    # 1. Optimization History (one per objective for multi-objective)
    for suffix, name, target in objectives:
        filename = (
            f"optimization_history_{suffix}.html"
            if is_multi_objective
            else "optimization_history.html"
        )
        print(f"  Generating {filename}...")
        try:
            if target:
                fig = plot_optimization_history(study, target=target, target_name=name)
            else:
                fig = plot_optimization_history(study)
            fig.write_html(output_dir / filename)
        except Exception as e:
            print(f"    Warning: Could not generate optimization history: {e}")

    # 2. Pareto Front (for multi-objective only)
    if is_multi_objective:
        print("  Generating pareto_front.html...")
        try:
            fig = plot_pareto_front(
                study,
                target_names=["KL Divergence", "Refusals"],
            )
            fig.write_html(output_dir / "pareto_front.html")
        except Exception as e:
            print(f"    Warning: Could not generate pareto front: {e}")

    # 3. Parameter Importances (one per objective for multi-objective)
    for suffix, name, target in objectives:
        filename = (
            f"param_importances_{suffix}.html"
            if is_multi_objective
            else "param_importances.html"
        )
        print(f"  Generating {filename}...")
        try:
            if target:
                fig = plot_param_importances(study, target=target, target_name=name)
            else:
                fig = plot_param_importances(study)
            fig.write_html(output_dir / filename)
        except Exception as e:
            print(f"    Warning: Could not generate param importances: {e}")

    # 4. Parallel Coordinate (one per objective for multi-objective)
    for suffix, name, target in objectives:
        filename = (
            f"parallel_coordinate_{suffix}.html"
            if is_multi_objective
            else "parallel_coordinate.html"
        )
        print(f"  Generating {filename}...")
        try:
            if target:
                fig = plot_parallel_coordinate(study, target=target, target_name=name)
            else:
                fig = plot_parallel_coordinate(study)
            fig.write_html(output_dir / filename)
        except Exception as e:
            print(f"    Warning: Could not generate parallel coordinate: {e}")

    # 5. Contour plots (one per objective for multi-objective)
    for suffix, name, target in objectives:
        filename = f"contour_{suffix}.html" if is_multi_objective else "contour.html"
        print(f"  Generating {filename}...")
        try:
            params = list(completed[0].params.keys()) if completed else []
            weight_params = [p for p in params if "max_weight" in p]
            if target:
                if len(weight_params) >= 2:
                    fig = plot_contour(
                        study, params=weight_params[:2], target=target, target_name=name
                    )
                else:
                    fig = plot_contour(study, target=target, target_name=name)
            else:
                if len(weight_params) >= 2:
                    fig = plot_contour(study, params=weight_params[:2])
                else:
                    fig = plot_contour(study)
            fig.write_html(output_dir / filename)
        except Exception as e:
            print(f"    Warning: Could not generate contour plot: {e}")

    # 6. Slice plots (one per objective for multi-objective)
    for suffix, name, target in objectives:
        filename = f"slice_{suffix}.html" if is_multi_objective else "slice.html"
        print(f"  Generating {filename}...")
        try:
            if target:
                fig = plot_slice(study, target=target, target_name=name)
            else:
                fig = plot_slice(study)
            fig.write_html(output_dir / filename)
        except Exception as e:
            print(f"    Warning: Could not generate slice plot: {e}")

    # 7. Timeline (same for all objectives)
    print("  Generating timeline.html...")
    try:
        fig = plot_timeline(study)
        fig.write_html(output_dir / "timeline.html")
    except Exception as e:
        print(f"    Warning: Could not generate timeline: {e}")

    print("\nDone! Open the HTML files in a browser to view the plots.")
    if is_multi_objective:
        print("\nQuick view:")
        print(f"  open {output_dir / 'pareto_front.html'}  (trade-off view)")
        print(f"  open {output_dir / 'optimization_history_kl.html'}  (KL progress)")
        print(
            f"  open {output_dir / 'optimization_history_refusals.html'}  (refusals progress)"
        )
    else:
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
        default="sqlite:///bruno_study.db",
        help="Optuna storage URL (default: sqlite:///bruno_study.db)",
    )
    parser.add_argument(
        "--study",
        dest="study_name",
        default="bruno_study",
        help="Study name (default: bruno_study)",
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
