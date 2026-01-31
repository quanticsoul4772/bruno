"""Bruno Monitor - Real-time visualization dashboard for abliteration progress.

A Gradio-based web dashboard that provides:
- Real-time trial progress visualization
- Interactive Plotly charts (optimization history, Pareto front, etc.)
- Trial comparison and parameter exploration
- Auto-refresh every 30 seconds

Usage:
    python examples/monitor_app.py                              # Use defaults
    python examples/monitor_app.py --storage sqlite:///my.db    # Custom storage
    python examples/monitor_app.py --study qwen32b-v2           # Specific study
    python examples/monitor_app.py --refresh 60                 # 60s refresh interval
"""

import argparse
import logging
import sys
import time
from typing import Optional

try:
    import gradio as gr
except ImportError:
    print("Error: gradio is required for the monitor dashboard.")
    print("Install with: pip install gradio")
    sys.exit(1)

try:
    import optuna
    import plotly.graph_objects as go
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_pareto_front,
        plot_timeline,
    )
except ImportError:
    print("Error: optuna with plotly is required.")
    print("Install with: pip install 'optuna[plotly]'")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bruno_monitor")

# Global configuration
STORAGE_URL: str = "sqlite:///heretic_study.db"
STUDY_NAME: str = "heretic_study"
REFRESH_INTERVAL: int = 30


class StudyMonitor:
    """Monitors an Optuna study and provides visualization data."""

    def __init__(self, storage: str, study_name: str) -> None:
        self.storage = storage
        self.study_name = study_name
        self._study: Optional[optuna.Study] = None
        self._last_refresh: float = 0
        self._trial_count: int = 0

    def load_study(self) -> Optional[optuna.Study]:
        """Load or reload the study from storage."""
        try:
            self._study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
            )
            self._last_refresh = time.time()
            self._trial_count = len(self._study.trials)
            return self._study
        except KeyError:
            logger.warning(f"Study '{self.study_name}' not found")
            return None
        except Exception as e:
            logger.error(f"Error loading study: {e}")
            return None

    def get_study(self, force_refresh: bool = False) -> Optional[optuna.Study]:
        """Get the study, refreshing if needed."""
        if force_refresh or self._study is None:
            return self.load_study()
        return self._study

    def get_completed_trials(self) -> list:
        """Get list of completed trials."""
        study = self.get_study()
        if study is None:
            return []
        return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def get_summary_stats(self) -> dict:
        """Get summary statistics for display."""
        study = self.get_study(force_refresh=True)
        if study is None:
            return {
                "status": "No study found",
                "total_trials": 0,
                "completed_trials": 0,
                "running_trials": 0,
                "failed_trials": 0,
                "best_value": "N/A",
                "best_params": {},
            }

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        running = [
            t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
        ]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        # Get best trial info
        best_value = "N/A"
        best_params = {}
        if completed:
            try:
                best_trial = study.best_trial
                if len(study.directions) > 1:
                    # Multi-objective: show first objective
                    best_value = f"{best_trial.values}"
                else:
                    best_value = f"{best_trial.value:.4f}"
                best_params = best_trial.params
            except Exception:
                pass

        return {
            "status": "Running"
            if running
            else ("Complete" if completed else "Starting"),
            "total_trials": len(study.trials),
            "completed_trials": len(completed),
            "running_trials": len(running),
            "failed_trials": len(failed),
            "best_value": best_value,
            "best_params": best_params,
            "last_refresh": time.strftime("%H:%M:%S"),
        }


# Global monitor instance
monitor: Optional[StudyMonitor] = None


def get_monitor() -> StudyMonitor:
    """Get or create the global monitor."""
    global monitor
    if monitor is None:
        monitor = StudyMonitor(STORAGE_URL, STUDY_NAME)
    return monitor


# =============================================================================
# Visualization Functions
# =============================================================================


def create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
    )
    return fig


def get_optimization_history() -> go.Figure:
    """Generate optimization history plot."""
    m = get_monitor()
    study = m.get_study(force_refresh=True)

    if study is None:
        return create_empty_figure(f"Study '{STUDY_NAME}' not found")

    completed = m.get_completed_trials()
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        fig = plot_optimization_history(study)
        fig.update_layout(
            title="Optimization History",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating optimization history: {e}")
        return create_empty_figure(f"Error: {e}")


def get_pareto_front() -> go.Figure:
    """Generate Pareto front plot for multi-objective optimization."""
    m = get_monitor()
    study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    if len(study.directions) <= 1:
        return create_empty_figure("Single-objective study (no Pareto front)")

    completed = m.get_completed_trials()
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        fig = plot_pareto_front(study)
        fig.update_layout(
            title="Pareto Front (KL Divergence vs Refusals)",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating Pareto front: {e}")
        return create_empty_figure(f"Error: {e}")


def get_param_importances() -> go.Figure:
    """Generate parameter importances plot."""
    m = get_monitor()
    study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    completed = m.get_completed_trials()
    if len(completed) < 5:
        return create_empty_figure(
            f"Need at least 5 trials for importance (have {len(completed)})"
        )

    try:
        fig = plot_param_importances(study)
        fig.update_layout(
            title="Parameter Importances",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating param importances: {e}")
        return create_empty_figure(f"Error: {e}")


def get_parallel_coordinate() -> go.Figure:
    """Generate parallel coordinate plot."""
    m = get_monitor()
    study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    completed = m.get_completed_trials()
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        fig = plot_parallel_coordinate(study)
        fig.update_layout(
            title="Parallel Coordinate Plot",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating parallel coordinate: {e}")
        return create_empty_figure(f"Error: {e}")


def get_timeline() -> go.Figure:
    """Generate trial timeline plot."""
    m = get_monitor()
    study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    completed = m.get_completed_trials()
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        fig = plot_timeline(study)
        fig.update_layout(
            title="Trial Timeline",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating timeline: {e}")
        return create_empty_figure(f"Error: {e}")


def get_summary_text() -> str:
    """Get summary statistics as formatted text."""
    m = get_monitor()
    stats = m.get_summary_stats()

    lines = [
        f"**Status:** {stats['status']}",
        f"**Total Trials:** {stats['total_trials']}",
        f"**Completed:** {stats['completed_trials']}",
        f"**Running:** {stats['running_trials']}",
        f"**Failed:** {stats['failed_trials']}",
        f"**Best Value:** {stats['best_value']}",
        f"\n**Last Refresh:** {stats.get('last_refresh', 'N/A')}",
    ]

    if stats["best_params"]:
        lines.append("\n**Best Parameters:**")
        for k, v in stats["best_params"].items():
            if isinstance(v, float):
                lines.append(f"  • {k}: {v:.4f}")
            else:
                lines.append(f"  • {k}: {v}")

    return "\n".join(lines)


def get_trials_table() -> list[list]:
    """Get trials as a table for display."""
    m = get_monitor()
    study = m.get_study()

    if study is None:
        return []

    rows = []
    for trial in reversed(study.trials[-50:]):  # Last 50 trials, newest first
        state = trial.state.name
        if trial.values:
            values = ", ".join(f"{v:.4f}" for v in trial.values)
        elif trial.value is not None:
            values = f"{trial.value:.4f}"
        else:
            values = "N/A"

        # Key parameters (first 3)
        params = ", ".join(
            f"{k.split('.')[-1]}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(trial.params.items())[:3]
        )

        rows.append([trial.number, state, values, params])

    return rows


def refresh_all():
    """Refresh all plots and data."""
    return (
        get_summary_text(),
        get_optimization_history(),
        get_pareto_front(),
        get_param_importances(),
        get_parallel_coordinate(),
        get_timeline(),
        get_trials_table(),
    )


def list_available_studies() -> str:
    """List all available studies in the storage."""
    try:
        summaries = optuna.study.get_all_study_summaries(STORAGE_URL)
        if not summaries:
            return "No studies found in storage."

        lines = ["**Available Studies:**\n"]
        for s in summaries:
            lines.append(f"• **{s.study_name}**: {s.n_trials} trials")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing studies: {e}"


def change_study(study_name: str) -> tuple:
    """Change the monitored study."""
    global monitor, STUDY_NAME
    STUDY_NAME = study_name
    monitor = StudyMonitor(STORAGE_URL, study_name)
    return refresh_all()


# =============================================================================
# Gradio UI
# =============================================================================


def create_ui() -> gr.Blocks:
    """Create the Gradio monitoring interface."""
    with gr.Blocks(title="Bruno Monitor") as demo:
        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 20px; border-bottom: 1px solid #e5e5e5;">
                <h1 style="margin: 0; font-size: 1.75em;">Bruno Monitor</h1>
                <p style="margin: 6px 0 0 0; color: #737373;">Real-time Abliteration Progress Dashboard</p>
            </div>
        """)

        with gr.Row():
            # Left sidebar - Summary
            with gr.Column(scale=1):
                gr.Markdown("### Study Info")
                summary_md = gr.Markdown(get_summary_text())

                refresh_btn = gr.Button("🔄 Refresh Now", variant="primary")

                with gr.Accordion("Study Selection", open=False):
                    gr.Markdown(list_available_studies())
                    study_name_input = gr.Textbox(
                        label="Study Name",
                        value=STUDY_NAME,
                        placeholder="Enter study name",
                    )
                    change_study_btn = gr.Button("Load Study", size="sm")

                with gr.Accordion("Configuration", open=False):
                    gr.Markdown(f"**Storage:** `{STORAGE_URL}`")
                    gr.Markdown(f"**Auto-refresh:** Every {REFRESH_INTERVAL}s")

            # Main content - Plots
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("📈 Progress"):
                        history_plot = gr.Plot(
                            label="Optimization History",
                            value=get_optimization_history(),
                        )

                    with gr.TabItem("🎯 Pareto Front"):
                        pareto_plot = gr.Plot(
                            label="Pareto Front",
                            value=get_pareto_front(),
                        )

                    with gr.TabItem("📊 Importance"):
                        importance_plot = gr.Plot(
                            label="Parameter Importances",
                            value=get_param_importances(),
                        )

                    with gr.TabItem("🔀 Parameters"):
                        parallel_plot = gr.Plot(
                            label="Parallel Coordinate",
                            value=get_parallel_coordinate(),
                        )

                    with gr.TabItem("⏱️ Timeline"):
                        timeline_plot = gr.Plot(
                            label="Trial Timeline",
                            value=get_timeline(),
                        )

                    with gr.TabItem("📋 Trials"):
                        trials_table = gr.Dataframe(
                            headers=["Trial", "State", "Values", "Key Parameters"],
                            value=get_trials_table(),
                            wrap=True,
                        )

        # Wire up events
        refresh_btn.click(
            refresh_all,
            outputs=[
                summary_md,
                history_plot,
                pareto_plot,
                importance_plot,
                parallel_plot,
                timeline_plot,
                trials_table,
            ],
        )

        change_study_btn.click(
            change_study,
            inputs=[study_name_input],
            outputs=[
                summary_md,
                history_plot,
                pareto_plot,
                importance_plot,
                parallel_plot,
                timeline_plot,
                trials_table,
            ],
        )

        # Auto-refresh using Gradio's built-in timer
        # Note: This creates a timer that fires every REFRESH_INTERVAL seconds
        demo.load(
            refresh_all,
            outputs=[
                summary_md,
                history_plot,
                pareto_plot,
                importance_plot,
                parallel_plot,
                timeline_plot,
                trials_table,
            ],
            every=REFRESH_INTERVAL,
        )

    return demo


def main() -> None:
    """Main entry point."""
    global STORAGE_URL, STUDY_NAME, REFRESH_INTERVAL, monitor

    parser = argparse.ArgumentParser(
        description="Bruno Monitor - Real-time abliteration visualization dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--storage",
        "-s",
        default="sqlite:///heretic_study.db",
        help="Optuna storage URL (default: sqlite:///heretic_study.db)",
    )
    parser.add_argument(
        "--study",
        default="heretic_study",
        help="Study name to monitor (default: heretic_study)",
    )
    parser.add_argument(
        "--refresh",
        "-r",
        type=int,
        default=30,
        help="Auto-refresh interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=7861,
        help="Port to run the dashboard on (default: 7861)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )

    args = parser.parse_args()

    STORAGE_URL = args.storage
    STUDY_NAME = args.study
    REFRESH_INTERVAL = args.refresh

    # Initialize monitor
    monitor = StudyMonitor(STORAGE_URL, STUDY_NAME)

    logger.info("Bruno Monitor - Starting...")
    logger.info(f"Storage: {STORAGE_URL}")
    logger.info(f"Study: {STUDY_NAME}")
    logger.info(f"Refresh interval: {REFRESH_INTERVAL}s")

    # Check if study exists
    if monitor.load_study() is None:
        logger.warning(f"Study '{STUDY_NAME}' not found in {STORAGE_URL}")
        logger.info("The dashboard will show available studies once data is available.")
    else:
        stats = monitor.get_summary_stats()
        logger.info(f"Loaded study with {stats['total_trials']} trials")

    # Create and launch UI
    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
