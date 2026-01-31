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
TARGET_TRIALS: int = 200  # Default target for progress bar

# Custom CSS for professional styling
CUSTOM_CSS = """
/* Overall theme */
.gradio-container {
    max-width: 1400px !important;
}

/* Cross-platform monospace font stack */
:root {
    --mono-font: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', 'Liberation Mono', 'Menlo', monospace;
}

/* Header styling */
.dashboard-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    color: white;
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.dashboard-header h1 {
    margin: 0;
    font-size: 1.5em;
    font-weight: 600;
}
.dashboard-header .study-name {
    background: rgba(255,255,255,0.2);
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.9em;
    margin-left: 12px;
}
.dashboard-header .subtitle {
    margin: 4px 0 0 0;
    opacity: 0.85;
    font-size: 0.9em;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    backdrop-filter: blur(2px);
}
.loading-spinner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}
.loading-spinner .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #e5e7eb;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
.loading-spinner .text {
    color: #4b5563;
    font-weight: 500;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: box-shadow 0.2s, transform 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.8em;
    font-weight: 700;
    font-family: var(--mono-font);
    margin: 4px 0;
}
.metric-label {
    font-size: 0.85em;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-sublabel {
    font-size: 0.75em;
    color: #9ca3af;
    margin-top: 2px;
}

/* Color semantics: green = good/low, amber = moderate, red = bad/high */
.color-good { color: #059669; }  /* Green - low values are good */
.color-warning { color: #d97706; }  /* Amber - moderate */
.color-bad { color: #dc2626; }  /* Red - high values are bad */
.color-neutral { color: #2563eb; }  /* Blue - neutral info */

/* Status badges */
.status-running {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #059669;
    background: #d1fae5;
    padding: 6px 14px;
    border-radius: 16px;
    font-weight: 600;
    font-size: 0.95em;
}
.status-running::before {
    content: '';
    width: 8px;
    height: 8px;
    background: #059669;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
.status-complete {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #2563eb;
    background: #dbeafe;
    padding: 6px 14px;
    border-radius: 16px;
    font-weight: 600;
    font-size: 0.95em;
}
.status-failed {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #dc2626;
    background: #fee2e2;
    padding: 6px 14px;
    border-radius: 16px;
    font-weight: 600;
    font-size: 0.95em;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Progress bar */
.progress-container {
    background: #f3f4f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
}
.progress-bar {
    background: #e5e7eb;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.progress-fill {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.progress-text {
    display: flex;
    justify-content: space-between;
    font-size: 0.85em;
    color: #6b7280;
    margin-top: 6px;
    font-family: var(--mono-font);
}

/* Auto-refresh indicator */
.refresh-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85em;
    color: #6b7280;
    padding: 8px 12px;
    background: #f9fafb;
    border-radius: 6px;
    margin-top: 8px;
}
.refresh-dot {
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
.refresh-dot.paused {
    background: #9ca3af;
    animation: none;
}

/* Best trial card */
.best-trial-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 12px;
}
.best-trial-card h4 {
    margin: 0 0 8px 0;
    color: #166534;
    font-size: 0.9em;
}
.best-trial-values {
    font-family: var(--mono-font);
    font-size: 0.95em;
}

/* Tab styling */
.tab-nav button {
    font-weight: 500 !important;
    transition: background 0.2s;
}

/* Table improvements */
.trials-table {
    font-family: var(--mono-font);
    font-size: 0.9em;
}

/* Plot transitions */
.plot-container {
    transition: opacity 0.3s ease;
}
.plot-loading {
    opacity: 0.5;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .dashboard-header {
        padding: 16px;
    }
    .dashboard-header h1 {
        font-size: 1.2em;
        flex-wrap: wrap;
    }
    .dashboard-header .study-name {
        margin-left: 0;
        margin-top: 8px;
        display: block;
        width: fit-content;
    }
    .metric-card {
        padding: 12px;
    }
    .metric-value {
        font-size: 1.4em;
    }
    .progress-container {
        padding: 10px 12px;
    }
}

@media (max-width: 480px) {
    .metric-grid {
        grid-template-columns: 1fr !important;
    }
    .refresh-indicator {
        flex-wrap: wrap;
    }
}
"""


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
                "status_class": "",
                "total_trials": 0,
                "completed_trials": 0,
                "running_trials": 0,
                "failed_trials": 0,
                "best_kl": "N/A",
                "best_refusals": "N/A",
                "best_params": {},
                "progress_pct": 0,
                "is_multi_objective": False,
            }

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        running = [
            t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
        ]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        # Determine status and CSS class
        if running:
            status = "Running"
            status_class = "status-running"
        elif len(completed) >= TARGET_TRIALS:
            status = "Complete"
            status_class = "status-complete"
        elif failed and not completed:
            status = "Failed"
            status_class = "status-failed"
        elif completed:
            status = "In Progress"
            status_class = "status-running"
        else:
            status = "Starting"
            status_class = ""

        # Get best trial info with proper formatting
        best_kl = "N/A"
        best_refusals = "N/A"
        best_params = {}
        is_multi_objective = len(study.directions) > 1

        if completed:
            try:
                # For multi-objective, find best by refusals (more interpretable)
                if is_multi_objective:
                    # Get Pareto front trials
                    best_trials = study.best_trials
                    if best_trials:
                        # Pick one with lowest refusals
                        best_trial = min(
                            best_trials,
                            key=lambda t: t.values[1] if t.values else float("inf"),
                        )
                        best_kl = f"{best_trial.values[0]:.3f}"
                        best_refusals = f"{int(best_trial.values[1])}/100"
                        best_params = best_trial.params
                else:
                    best_trial = study.best_trial
                    best_kl = f"{best_trial.value:.4f}"
                    best_params = best_trial.params
            except Exception:
                pass

        # Calculate progress percentage
        progress_pct = min(100, (len(completed) / TARGET_TRIALS) * 100)

        return {
            "status": status,
            "status_class": status_class,
            "total_trials": len(study.trials),
            "completed_trials": len(completed),
            "running_trials": len(running),
            "failed_trials": len(failed),
            "best_kl": best_kl,
            "best_refusals": best_refusals,
            "best_params": best_params,
            "progress_pct": progress_pct,
            "is_multi_objective": is_multi_objective,
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


def get_optimization_history(study: Optional[optuna.Study] = None) -> go.Figure:
    """Generate optimization history plot.

    Args:
        study: Optional pre-loaded study to avoid redundant DB calls.
    """
    if study is None:
        m = get_monitor()
        study = m.get_study(force_refresh=True)

    if study is None:
        return create_empty_figure(f"Study '{STUDY_NAME}' not found")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        # For multi-objective, plot KL divergence (first objective)
        if len(study.directions) > 1:
            def target(t):
                return t.values[0] if t.values else float("inf")
            fig = plot_optimization_history(
                study, target=target, target_name="KL Divergence"
            )
        else:
            fig = plot_optimization_history(study)
        fig.update_layout(
            title="Optimization History (KL Divergence)",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating optimization history: {e}")
        return create_empty_figure(f"Error: {e}")


def get_pareto_front(study: Optional[optuna.Study] = None) -> go.Figure:
    """Generate Pareto front plot for multi-objective optimization.

    Args:
        study: Optional pre-loaded study to avoid redundant DB calls.
    """
    if study is None:
        m = get_monitor()
        study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    if len(study.directions) <= 1:
        return create_empty_figure("Single-objective study (no Pareto front)")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        fig = plot_pareto_front(
            study,
            target_names=["KL Divergence", "Refusals"],
        )
        fig.update_layout(
            title="Pareto Front (KL Divergence vs Refusals)",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating Pareto front: {e}")
        return create_empty_figure(f"Error: {e}")


def get_param_importances(study: Optional[optuna.Study] = None) -> go.Figure:
    """Generate parameter importances plot.

    Args:
        study: Optional pre-loaded study to avoid redundant DB calls.
    """
    if study is None:
        m = get_monitor()
        study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 5:
        return create_empty_figure(
            f"Need at least 5 trials for importance (have {len(completed)})"
        )

    try:
        # For multi-objective, plot importance for refusals (usually more interesting)
        if len(study.directions) > 1:
            def target(t):
                return (t.values[1]
                            if t.values and len(t.values) > 1
                            else float("inf"))
            fig = plot_param_importances(study, target=target, target_name="Refusals")
        else:
            fig = plot_param_importances(study)
        fig.update_layout(
            title="Parameter Importances (for Refusals)",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating param importances: {e}")
        return create_empty_figure(f"Error: {e}")


def get_parallel_coordinate(study: Optional[optuna.Study] = None) -> go.Figure:
    """Generate parallel coordinate plot.

    Args:
        study: Optional pre-loaded study to avoid redundant DB calls.
    """
    if study is None:
        m = get_monitor()
        study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return create_empty_figure(f"Need at least 2 trials (have {len(completed)})")

    try:
        # For multi-objective, plot for KL divergence
        if len(study.directions) > 1:
            def target(t):
                return t.values[0] if t.values else float("inf")
            fig = plot_parallel_coordinate(
                study, target=target, target_name="KL Divergence"
            )
        else:
            fig = plot_parallel_coordinate(study)
        fig.update_layout(
            title="Parallel Coordinate Plot",
            template="plotly_white",
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating parallel coordinate: {e}")
        return create_empty_figure(f"Error: {e}")


def get_timeline(study: Optional[optuna.Study] = None) -> go.Figure:
    """Generate trial timeline plot.

    Args:
        study: Optional pre-loaded study to avoid redundant DB calls.
    """
    if study is None:
        m = get_monitor()
        study = m.get_study()

    if study is None:
        return create_empty_figure("No study loaded")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
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


def get_header_html() -> str:
    """Generate the dashboard header with study name."""
    return f"""
    <div class="dashboard-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1>Bruno Monitor <span class="study-name">{STUDY_NAME}</span></h1>
                <p class="subtitle">Real-time Abliteration Progress Dashboard</p>
            </div>
        </div>
    </div>
    """


def get_status_html(stats: Optional[dict] = None) -> str:
    """Get status badge HTML.

    Args:
        stats: Optional pre-computed stats to avoid redundant calls.
    """
    if stats is None:
        m = get_monitor()
        stats = m.get_summary_stats()
    status = stats["status"]
    status_class = stats["status_class"]
    return f'<span class="{status_class}">{status}</span>'


def get_progress_html(stats: Optional[dict] = None) -> str:
    """Get progress bar HTML.

    Args:
        stats: Optional pre-computed stats to avoid redundant calls.
    """
    if stats is None:
        m = get_monitor()
        stats = m.get_summary_stats()
    completed = stats["completed_trials"]
    progress_pct = stats["progress_pct"]

    return f"""
    <div class="progress-container">
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_pct}%;"></div>
        </div>
        <div class="progress-text">
            <span>{completed} / {TARGET_TRIALS} trials</span>
            <span>{progress_pct:.1f}%</span>
        </div>
    </div>
    """


def get_metrics_html() -> str:
    """Get metric cards HTML."""
    m = get_monitor()
    stats = m.get_summary_stats()

    return f"""
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 12px 0;">
        <div class="metric-card">
            <div class="metric-label">Completed</div>
            <div class="metric-value" style="color: #2563eb;">{stats["completed_trials"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Running</div>
            <div class="metric-value" style="color: #059669;">{stats["running_trials"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best KL</div>
            <div class="metric-value" style="color: #7c3aed; font-size: 1.4em;">{stats["best_kl"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Refusals</div>
            <div class="metric-value" style="color: #dc2626; font-size: 1.4em;">{stats["best_refusals"]}</div>
        </div>
    </div>
    """


def get_best_params_html() -> str:
    """Get best parameters HTML."""
    m = get_monitor()
    stats = m.get_summary_stats()

    if not stats["best_params"]:
        return ""

    params_html = ""
    for k, v in list(stats["best_params"].items())[:4]:  # Show top 4 params
        short_name = k.split(".")[-1]
        if isinstance(v, float):
            params_html += f"<div><strong>{short_name}:</strong> {v:.3f}</div>"
        else:
            params_html += f"<div><strong>{short_name}:</strong> {v}</div>"

    return f"""
    <div class="best-trial-card">
        <h4>Best Trial Parameters</h4>
        <div class="best-trial-values">{params_html}</div>
    </div>
    """


def get_refresh_indicator_html(is_active: bool = True) -> str:
    """Get auto-refresh indicator HTML."""
    m = get_monitor()
    stats = m.get_summary_stats()
    dot_class = "refresh-dot" if is_active else "refresh-dot paused"
    status_text = (
        f"Auto-refresh: {REFRESH_INTERVAL}s" if is_active else "Auto-refresh paused"
    )

    return f"""
    <div class="refresh-indicator">
        <span class="{dot_class}"></span>
        <span>{status_text}</span>
        <span style="margin-left: auto;">Updated: {stats.get("last_refresh", "N/A")}</span>
    </div>
    """


def get_trials_table(study: Optional[optuna.Study] = None) -> list[list]:
    """Get trials as a table for display.

    Args:
        study: Optional pre-loaded study to avoid redundant DB calls.
    """
    if study is None:
        m = get_monitor()
        study = m.get_study()

    if study is None:
        return []

    is_multi_objective = len(study.directions) > 1
    rows = []
    for trial in reversed(study.trials[-50:]):  # Last 50 trials, newest first
        state = trial.state.name

        # Format values more clearly - don't hardcode /100
        if trial.values and is_multi_objective:
            kl = f"{trial.values[0]:.3f}"
            refusals = str(int(trial.values[1])) if len(trial.values) > 1 else "N/A"
        elif trial.value is not None:
            kl = f"{trial.value:.4f}"
            refusals = "N/A"
        else:
            kl = "N/A"
            refusals = "N/A"

        # Key parameters (first 3)
        params = ", ".join(
            f"{k.split('.')[-1]}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(trial.params.items())[:3]
        )

        rows.append([trial.number, state, kl, refusals, params])

    return rows


def refresh_all():
    """Refresh all plots and data.

    Loads the study ONCE and passes it to all functions to avoid
    redundant database calls (was loading 5+ times before).
    """
    m = get_monitor()  # Load study once and reuse for all operations
    study = m.get_study(force_refresh=True)

    return (
        get_header_html(),
        get_status_html(),
        get_progress_html(),
        get_metrics_html(),
        get_best_params_html(),
        get_refresh_indicator_html(True),
        get_optimization_history(study),
        get_pareto_front(study),
        get_param_importances(study),
        get_parallel_coordinate(study),
        get_timeline(study),
        get_trials_table(study),
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

    # Check if study exists and provide helpful error message
    m = get_monitor()
    study = m.get_study(force_refresh=True)
    if study is None:
        logger.warning(f"Study '{study_name}' not found")

    return refresh_all()


def toggle_auto_refresh(is_active: bool) -> tuple:
    """Toggle auto-refresh and update indicator."""
    return (
        get_refresh_indicator_html(is_active),
        gr.Timer(value=REFRESH_INTERVAL, active=is_active),
    )


# =============================================================================
# Gradio UI
# =============================================================================


def get_loading_html(visible: bool = False) -> str:
    """Get loading overlay HTML."""
    display = "flex" if visible else "none"
    return f"""
    <div class="loading-overlay" style="display: {display};">
        <div class="loading-spinner">
            <div class="spinner"></div>
            <div class="text">Refreshing...</div>
        </div>
    </div>
    """


def create_ui() -> gr.Blocks:
    """Create the Gradio monitoring interface."""
    with gr.Blocks(title="Bruno Monitor", css=CUSTOM_CSS) as demo:
        # Auto-refresh timer (Gradio 6.x compatible)
        timer = gr.Timer(value=REFRESH_INTERVAL, active=True)

        # Header with study name
        header_html = gr.HTML(get_header_html())

        with gr.Row():
            # Left sidebar - Summary (wider for better readability)
            with gr.Column(scale=1, min_width=320):
                # Status badge (no redundant "### Status" label)
                status_html = gr.HTML(get_status_html())

                # Progress bar
                progress_html = gr.HTML(get_progress_html())

                # Metric cards
                metrics_html = gr.HTML(get_metrics_html())

                # Best parameters
                best_params_html = gr.HTML(get_best_params_html())

                # Refresh controls
                with gr.Row():
                    refresh_btn = gr.Button(
                        "Refresh Now",
                        variant="primary",
                        scale=2,
                        size="sm",
                    )
                    auto_refresh_toggle = gr.Checkbox(
                        label="Auto-refresh",
                        value=True,
                        scale=1,
                    )

                # Auto-refresh indicator
                refresh_indicator_html = gr.HTML(get_refresh_indicator_html(True))

                # Study selection (collapsed by default)
                with gr.Accordion("Change Study", open=False):
                    gr.Markdown(list_available_studies())
                    study_name_input = gr.Textbox(
                        label="Study Name",
                        value=STUDY_NAME,
                        placeholder="Enter study name",
                        show_label=False,
                    )
                    change_study_btn = gr.Button(
                        "Load Study", size="sm", variant="secondary"
                    )

            # Main content - Plots
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Progress", id="progress"):
                        history_plot = gr.Plot(
                            label="Optimization History",
                            value=get_optimization_history(),
                        )

                    with gr.TabItem("Pareto Front", id="pareto"):
                        pareto_plot = gr.Plot(
                            label="Pareto Front",
                            value=get_pareto_front(),
                        )

                    with gr.TabItem("Importance", id="importance"):
                        importance_plot = gr.Plot(
                            label="Parameter Importances",
                            value=get_param_importances(),
                        )

                    with gr.TabItem("Parameters", id="params"):
                        parallel_plot = gr.Plot(
                            label="Parallel Coordinate",
                            value=get_parallel_coordinate(),
                        )

                    with gr.TabItem("Timeline", id="timeline"):
                        timeline_plot = gr.Plot(
                            label="Trial Timeline",
                            value=get_timeline(),
                        )

                    with gr.TabItem("Trials", id="trials"):
                        trials_table = gr.Dataframe(
                            headers=[
                                "Trial",
                                "State",
                                "KL Div",
                                "Refusals",
                                "Key Parameters",
                            ],
                            value=get_trials_table(),
                            wrap=True,
                            elem_classes=["trials-table"],
                        )

        # All outputs for refresh
        all_outputs = [
            header_html,
            status_html,
            progress_html,
            metrics_html,
            best_params_html,
            refresh_indicator_html,
            history_plot,
            pareto_plot,
            importance_plot,
            parallel_plot,
            timeline_plot,
            trials_table,
        ]

        # Wire up events
        refresh_btn.click(
            refresh_all,
            outputs=all_outputs,
        )

        change_study_btn.click(
            change_study,
            inputs=[study_name_input],
            outputs=all_outputs,
        )

        # Auto-refresh using gr.Timer (Gradio 6.x compatible)
        timer.tick(
            refresh_all,
            outputs=all_outputs,
        )

        # Toggle auto-refresh on/off with visual feedback
        auto_refresh_toggle.change(
            toggle_auto_refresh,
            inputs=[auto_refresh_toggle],
            outputs=[refresh_indicator_html, timer],
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
