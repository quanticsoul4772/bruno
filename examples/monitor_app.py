"""Bruno Monitor - Real-time visualization dashboard for abliteration progress.

A Gradio-based web dashboard that provides:
- Real-time trial progress visualization
- Time estimates (elapsed, ETA, avg per trial)
- Model and GPU context display
- Interactive Plotly charts (optimization history, Pareto front, etc.)
- Trial comparison and parameter exploration
- Convergence indicators
- Auto-refresh every 30 seconds

Usage:
    python examples/monitor_app.py                              # Use defaults
    python examples/monitor_app.py --storage sqlite:///my.db    # Custom storage
    python examples/monitor_app.py --study qwen32b-v2           # Specific study
    python examples/monitor_app.py --refresh 60                 # 60s refresh interval
    python examples/monitor_app.py --target-trials 300          # Custom trial target
    python examples/monitor_app.py --model "Moonlight-16B"      # Show model name
    python examples/monitor_app.py --gpu "H200 141GB"           # Show GPU info
"""

import argparse
import logging
import sys
import time
from datetime import timedelta
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
TOTAL_PROMPTS: int = 104  # Total prompts for refusal counting
MODEL_NAME: str = ""  # Model being abliterated
GPU_INFO: str = ""  # GPU info (e.g., "H200 141GB")

# Estimated baseline refusal rate (typical models refuse ~92% of bad prompts initially)
# Used as fallback when user_attrs["refusals"] is not available
ESTIMATED_BASELINE_RATE: float = 0.92


def get_trial_refusal_count(
    trial: "optuna.trial.FrozenTrial", total_prompts: int = TOTAL_PROMPTS
) -> int:
    """Get refusal count from trial, preferring user_attrs over ratio estimation.

    Bruno stores raw refusal counts in trial.user_attrs["refusals"], but older
    trials may only have trial.values[1] which is refusal_ratio (refusals/base_refusals).

    Args:
        trial: Optuna trial object (FrozenTrial from completed trials)
        total_prompts: Total number of prompts for percentage calculation

    Returns:
        Estimated refusal count (integer)
    """
    if "refusals" in trial.user_attrs:
        return int(trial.user_attrs["refusals"])
    # Fallback: values[1] is ratio relative to baseline
    if trial.values and len(trial.values) > 1:
        ratio = trial.values[1]
        estimated_baseline = int(total_prompts * ESTIMATED_BASELINE_RATE)
        return min(int(ratio * estimated_baseline), total_prompts)
    return total_prompts


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

/* Header styling - compact */
.dashboard-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    color: white;
    padding: 10px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.dashboard-header h1 {
    margin: 0;
    font-size: 1.2em;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dashboard-header .study-name {
    background: rgba(255,255,255,0.2);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8em;
}
.dashboard-header .subtitle {
    display: none;
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

/* Metric cards - compact horizontal */
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 8px 12px;
    text-align: center;
    transition: box-shadow 0.2s;
    min-width: 80px;
}
.metric-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.metric-value {
    font-size: 1.3em;
    font-weight: 700;
    font-family: var(--mono-font);
    margin: 2px 0;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.7em;
    color: #111827 !important;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    font-weight: 700;
}
.metric-sublabel {
    font-size: 0.65em;
    color: #374151 !important;
    margin-top: 1px;
    font-weight: 500;
}

/* Metrics row - horizontal bar */
.metrics-row {
    display: flex;
    gap: 8px;
    margin: 8px 0;
    flex-wrap: wrap;
    justify-content: space-between;
}

/* Color semantics: green = good/low, amber = moderate, red = bad/high */
.color-good { color: #059669; }  /* Green - low values are good */
.color-warning { color: #d97706; }  /* Amber - moderate */
.color-bad { color: #dc2626; }  /* Red - high values are bad */
.color-neutral { color: #2563eb; }  /* Blue - neutral info */

/* Status badges - compact */
.status-running {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #059669;
    background: #d1fae5;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.8em;
}
.status-running::before {
    content: '';
    width: 6px;
    height: 6px;
    background: #059669;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
.status-complete {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #2563eb;
    background: #dbeafe;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.8em;
}
.status-failed {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #dc2626;
    background: #fee2e2;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.8em;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Progress bar - compact full width */
.progress-container {
    background: #f3f4f6;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 4px 0 8px 0;
}
.progress-bar {
    background: #e5e7eb;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.progress-fill {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}
.progress-text {
    display: flex;
    justify-content: space-between;
    font-size: 0.75em;
    color: #111827 !important;
    margin-top: 4px;
    font-family: var(--mono-font);
    font-weight: 600;
}
.progress-text span {
    color: #111827 !important;
}

/* Auto-refresh indicator - compact */
.refresh-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7em;
    color: #111827 !important;
    padding: 4px 8px;
    background: #e5e7eb;
    border-radius: 4px;
    margin-top: 4px;
    font-weight: 600;
}
.refresh-indicator span {
    color: #111827 !important;
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

/* Best trial card - compact inline */
.best-trial-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 6px 10px;
    margin-top: 6px;
}
.best-trial-card h4 {
    margin: 0 0 4px 0;
    color: #14532d !important;
    font-size: 0.75em;
    font-weight: 700;
}
.best-trial-values {
    font-family: var(--mono-font);
    font-size: 0.7em;
    color: #14532d !important;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.best-trial-values div {
    color: #14532d !important;
    white-space: nowrap;
}
.best-trial-values strong {
    color: #052e16 !important;
    font-weight: 700;
}

/* Time estimates - hidden, merged into metrics */
.time-card {
    display: none;
}

/* Convergence indicator - compact */
.convergence-improving {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #059669;
    background: #d1fae5;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.7em;
}
.convergence-plateauing {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #d97706;
    background: #fef3c7;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.7em;
}
.convergence-converged {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #2563eb;
    background: #dbeafe;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.7em;
}

/* Model info in header - inline */
.model-info {
    display: flex;
    gap: 8px;
    margin: 0;
    flex-wrap: wrap;
}
.model-badge {
    background: rgba(255,255,255,0.15);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75em;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}
.model-badge .icon {
    opacity: 0.8;
    font-size: 0.9em;
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


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 0:
        return "N/A"
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


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
                "best_refusals_pct": "N/A",
                "best_params": {},
                "progress_pct": 0,
                "is_multi_objective": False,
                "elapsed": "N/A",
                "eta": "N/A",
                "avg_trial_time": "N/A",
                "convergence": "starting",
                "convergence_class": "",
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
        best_refusals_pct = "N/A"
        best_refusals_raw = None
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
                        best_trial = min(best_trials, key=get_trial_refusal_count)
                        best_kl = f"{best_trial.values[0]:.3f}"

                        # Get refusals from user_attrs (raw count) or estimate from ratio
                        best_refusals_raw = get_trial_refusal_count(best_trial)
                        best_refusals = f"{best_refusals_raw}/{TOTAL_PROMPTS}"
                        best_refusals_pct = (
                            f"{(best_refusals_raw / TOTAL_PROMPTS) * 100:.1f}%"
                        )
                        best_params = best_trial.params
                else:
                    best_trial = study.best_trial
                    best_kl = f"{best_trial.value:.4f}"
                    best_params = best_trial.params
            except Exception as e:
                logger.warning(f"Error getting best trial: {e}")
                pass

        # Calculate progress percentage
        progress_pct = min(100, (len(completed) / TARGET_TRIALS) * 100)

        # Calculate time estimates
        elapsed = "N/A"
        eta = "N/A"
        avg_trial_time = "N/A"

        if len(completed) >= 2:
            try:
                # Get timestamps from completed trials
                start_times = [t.datetime_start for t in completed if t.datetime_start]
                complete_times = [
                    t.datetime_complete for t in completed if t.datetime_complete
                ]

                if start_times and complete_times:
                    first_start = min(start_times)
                    last_complete = max(complete_times)
                    elapsed_td = last_complete - first_start
                    elapsed_secs = elapsed_td.total_seconds()
                    elapsed = format_duration(elapsed_secs)

                    # Average time per trial
                    avg_secs = elapsed_secs / len(completed)
                    avg_trial_time = f"{avg_secs:.1f}s"

                    # ETA for remaining trials
                    remaining_trials = TARGET_TRIALS - len(completed)
                    if remaining_trials > 0:
                        eta_secs = avg_secs * remaining_trials
                        eta = format_duration(eta_secs)
                    else:
                        eta = "Complete"
            except Exception:
                pass

        # Calculate convergence indicator
        # Logic:
        # - "Improving" = refusals dropping significantly over trials
        # - "Exploring" = refusals still high (>50%), optimization searching
        # - "Plateauing" = refusals low but not improving much
        # - "Converged" = refusals low (<20%) AND stable
        convergence = "starting"
        convergence_class = ""

        if len(completed) >= 20:
            try:
                sorted_trials = sorted(completed, key=lambda t: t.number)
                early_trials = sorted_trials[:10]
                recent_trials = sorted_trials[-10:]

                if is_multi_objective:
                    # Get best refusals from each group
                    early_best_count = min(
                        get_trial_refusal_count(t) for t in early_trials
                    )
                    recent_best_count = min(
                        get_trial_refusal_count(t) for t in recent_trials
                    )

                    # Convert to percentage of total prompts
                    early_pct = (early_best_count / TOTAL_PROMPTS) * 100
                    recent_pct = (recent_best_count / TOTAL_PROMPTS) * 100

                    improvement = early_pct - recent_pct  # Positive = getting better

                    # Decision logic based on ABSOLUTE refusal rate + improvement
                    if recent_pct > 50:  # Still refusing most prompts
                        if improvement > 5:
                            convergence = "Improving"
                            convergence_class = "convergence-improving"
                        else:
                            convergence = "Exploring"  # Still searching for good params
                            convergence_class = (
                                "convergence-plateauing"  # Yellow - not good yet
                            )
                    elif recent_pct > 20:  # Moderate refusals
                        if improvement > 3:
                            convergence = "Improving"
                            convergence_class = "convergence-improving"
                        else:
                            convergence = "Plateauing"
                            convergence_class = "convergence-plateauing"
                    else:  # Low refusals (<20%) - this is success territory
                        if improvement > 2:
                            convergence = "Improving"
                            convergence_class = "convergence-improving"
                        else:
                            convergence = "Converged"
                            convergence_class = "convergence-converged"
            except Exception:
                pass

        return {
            "status": status,
            "status_class": status_class,
            "total_trials": len(study.trials),
            "completed_trials": len(completed),
            "running_trials": len(running),
            "failed_trials": len(failed),
            "best_kl": best_kl,
            "best_refusals": best_refusals,
            "best_refusals_pct": best_refusals_pct,
            "best_params": best_params,
            "progress_pct": progress_pct,
            "is_multi_objective": is_multi_objective,
            "last_refresh": time.strftime("%H:%M:%S"),
            "elapsed": elapsed,
            "eta": eta,
            "avg_trial_time": avg_trial_time,
            "convergence": convergence,
            "convergence_class": convergence_class,
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
                return t.values[1] if t.values and len(t.values) > 1 else float("inf")

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
    """Generate performance over time chart showing KL and refusals improvement.

    Shows:
    - Individual trial KL values (scatter)
    - Individual trial refusal rates (scatter)
    - Running best KL line (shows improvement trajectory)
    - Running best refusals line (shows improvement trajectory)

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

    is_multi_objective = len(study.directions) > 1
    if not is_multi_objective:
        # Fall back to default timeline for single-objective
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

    try:
        # Sort trials by number
        sorted_trials = sorted(completed, key=lambda t: t.number)

        trial_numbers = []
        kl_values = []
        refusal_values = []

        for t in sorted_trials:
            if t.values and len(t.values) >= 2:
                trial_numbers.append(t.number)
                kl_values.append(t.values[0])
                # Get refusal count using shared helper
                refusal_count = get_trial_refusal_count(t)
                refusal_pct = (refusal_count / TOTAL_PROMPTS) * 100
                refusal_values.append(refusal_pct)

        if len(trial_numbers) < 2:
            return create_empty_figure("Not enough valid trials")

        # Calculate running best (minimum so far)
        running_best_kl = []
        running_best_refusals = []
        best_kl_so_far = float("inf")
        best_ref_so_far = float("inf")

        for kl, ref in zip(kl_values, refusal_values):
            best_kl_so_far = min(best_kl_so_far, kl)
            best_ref_so_far = min(best_ref_so_far, ref)
            running_best_kl.append(best_kl_so_far)
            running_best_refusals.append(best_ref_so_far)

        # Create figure with secondary y-axis
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Individual KL values (scatter, left y-axis)
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=kl_values,
                mode="markers",
                name="KL Divergence",
                marker=dict(color="#7c3aed", size=6, opacity=0.4),
                hovertemplate="Trial %{x}<br>KL: %{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Running best KL line (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=running_best_kl,
                mode="lines",
                name="Best KL",
                line=dict(color="#7c3aed", width=3),
                hovertemplate="Trial %{x}<br>Best KL: %{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Individual refusal values (scatter, right y-axis)
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=refusal_values,
                mode="markers",
                name="Refusal %",
                marker=dict(color="#dc2626", size=6, opacity=0.4),
                hovertemplate="Trial %{x}<br>Refusals: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        # Running best refusals line (right y-axis)
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=running_best_refusals,
                mode="lines",
                name="Best Refusal %",
                line=dict(color="#dc2626", width=3),
                hovertemplate="Trial %{x}<br>Best Refusals: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title="Performance Over Time",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(title_text="Trial Number")
        fig.update_yaxes(title_text="KL Divergence", secondary_y=False, color="#7c3aed")
        fig.update_yaxes(
            title_text="Refusal Rate (%)", secondary_y=True, color="#dc2626"
        )

        return fig
    except Exception as e:
        logger.error(f"Error generating performance timeline: {e}")
        return create_empty_figure(f"Error: {e}")


def get_header_html() -> str:
    """Generate the dashboard header with study name, model, and GPU info."""
    model_badge = ""
    gpu_badge = ""

    if MODEL_NAME:
        model_badge = f'<span class="model-badge">{MODEL_NAME}</span>'
    if GPU_INFO:
        gpu_badge = f'<span class="model-badge">{GPU_INFO}</span>'

    model_info = ""
    if model_badge or gpu_badge:
        model_info = f'<div class="model-info">{model_badge}{gpu_badge}</div>'

    return f"""
    <div class="dashboard-header">
        <h1>Bruno Monitor <span class="study-name">{STUDY_NAME}</span></h1>
        {model_info}
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
    """Get metric cards HTML - horizontal row."""
    m = get_monitor()
    stats = m.get_summary_stats()

    # Convergence badge
    convergence_html = ""
    if stats["convergence"] and stats["convergence"] != "starting":
        convergence_html = (
            f'<span class="{stats["convergence_class"]}">{stats["convergence"]}</span>'
        )

    return f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">Trials</div>
            <div class="metric-value" style="color: #2563eb;">{stats["completed_trials"]}/{TARGET_TRIALS}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ETA</div>
            <div class="metric-value" style="color: #059669;">{stats["eta"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Elapsed</div>
            <div class="metric-value" style="color: #6366f1;">{stats["elapsed"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best KL</div>
            <div class="metric-value" style="color: #7c3aed;">{stats["best_kl"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Refusals</div>
            <div class="metric-value" style="color: #dc2626;">{stats["best_refusals"]}</div>
            <div class="metric-sublabel">{stats["best_refusals_pct"]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Per Trial</div>
            <div class="metric-value" style="color: #0891b2;">{stats["avg_trial_time"]}</div>
        </div>
        <div class="metric-card" style="display: flex; align-items: center; justify-content: center;">
            {convergence_html}
        </div>
    </div>
    """


def get_time_estimates_html() -> str:
    """Get time estimates card HTML."""
    m = get_monitor()
    stats = m.get_summary_stats()

    return f"""
    <div class="time-card">
        <h4>Time Estimates</h4>
        <div class="time-values">
            <div>
                <div class="label">Elapsed</div>
                <div class="value">{stats["elapsed"]}</div>
            </div>
            <div>
                <div class="label">Remaining</div>
                <div class="value">{stats["eta"]}</div>
            </div>
            <div>
                <div class="label">Per Trial</div>
                <div class="value">{stats["avg_trial_time"]}</div>
            </div>
            <div>
                <div class="label">Failed</div>
                <div class="value">{stats["failed_trials"]}</div>
            </div>
        </div>
    </div>
    """


def get_best_params_html() -> str:
    """Get best parameters HTML - compact inline."""
    m = get_monitor()
    stats = m.get_summary_stats()

    if not stats["best_params"]:
        return ""

    params_html = ""
    for k, v in list(stats["best_params"].items())[:6]:  # Show top 6 params inline
        short_name = k.split(".")[-1]
        if isinstance(v, float):
            params_html += f"<div><strong>{short_name}:</strong> {v:.2f}</div>"
        else:
            params_html += f"<div><strong>{short_name}:</strong> {v}</div>"

    return f"""
    <div class="best-trial-card">
        <h4>Best Parameters</h4>
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

        # Format values - show refusals as count and percentage
        if trial.values and is_multi_objective:
            kl = f"{trial.values[0]:.3f}"
            if len(trial.values) > 1:
                # Get refusal count using shared helper
                refusal_count = get_trial_refusal_count(trial)
                refusal_pct = (refusal_count / TOTAL_PROMPTS) * 100
                refusals = f"{refusal_count} ({refusal_pct:.0f}%)"
            else:
                refusals = "N/A"
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
        get_metrics_html(),
        get_progress_html(),
        get_status_html(),
        get_best_params_html(),
        "",  # time_estimates_html - hidden
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

        # Compact header
        header_html = gr.HTML(get_header_html())

        # Full-width metrics row at top
        metrics_html = gr.HTML(get_metrics_html())

        # Full-width progress bar
        progress_html = gr.HTML(get_progress_html())

        with gr.Row():
            # Left sidebar - narrow, just controls and best params
            with gr.Column(scale=1, min_width=200):
                # Status badge
                status_html = gr.HTML(get_status_html())

                # Best parameters (compact)
                best_params_html = gr.HTML(get_best_params_html())

                # Time estimates - hidden, info merged into metrics
                time_estimates_html = gr.HTML("")

                # Refresh controls - compact
                with gr.Row():
                    refresh_btn = gr.Button(
                        "Refresh",
                        variant="primary",
                        scale=2,
                        size="sm",
                    )
                    auto_refresh_toggle = gr.Checkbox(
                        label="Auto",
                        value=True,
                        scale=1,
                    )

                # Auto-refresh indicator
                refresh_indicator_html = gr.HTML(get_refresh_indicator_html(True))

                # Study selection (collapsed)
                with gr.Accordion("Study", open=False):
                    gr.Markdown(list_available_studies())
                    study_name_input = gr.Textbox(
                        label="Study Name",
                        value=STUDY_NAME,
                        placeholder="Study name",
                        show_label=False,
                    )
                    change_study_btn = gr.Button("Load", size="sm", variant="secondary")

            # Main content - Plots (larger)
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.TabItem("Progress", id="progress"):
                        history_plot = gr.Plot(
                            label="Optimization History",
                            value=get_optimization_history(),
                        )

                    with gr.TabItem("Pareto", id="pareto"):
                        pareto_plot = gr.Plot(
                            label="Pareto Front",
                            value=get_pareto_front(),
                        )

                    with gr.TabItem("Importance", id="importance"):
                        importance_plot = gr.Plot(
                            label="Parameter Importances",
                            value=get_param_importances(),
                        )

                    with gr.TabItem("Params", id="params"):
                        parallel_plot = gr.Plot(
                            label="Parallel Coordinate",
                            value=get_parallel_coordinate(),
                        )

                    with gr.TabItem("Performance", id="timeline"):
                        timeline_plot = gr.Plot(
                            label="Performance Over Time",
                            value=get_timeline(),
                        )

                    with gr.TabItem("Trials", id="trials"):
                        trials_table = gr.Dataframe(
                            headers=[
                                "#",
                                "State",
                                "KL",
                                "Refusals",
                                "Params",
                            ],
                            value=get_trials_table(),
                            wrap=True,
                            elem_classes=["trials-table"],
                        )

        # All outputs for refresh
        all_outputs = [
            header_html,
            metrics_html,
            progress_html,
            status_html,
            best_params_html,
            time_estimates_html,
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
    global \
        STORAGE_URL, \
        STUDY_NAME, \
        REFRESH_INTERVAL, \
        TARGET_TRIALS, \
        TOTAL_PROMPTS, \
        MODEL_NAME, \
        GPU_INFO, \
        monitor

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
    parser.add_argument(
        "--target-trials",
        "-t",
        type=int,
        default=200,
        help="Target number of trials for progress bar (default: 200)",
    )
    parser.add_argument(
        "--total-prompts",
        type=int,
        default=104,
        help="Total prompts for refusal calculation (default: 104)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        help="Model name to display in header (e.g., 'Moonlight-16B-A3B-Instruct')",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        default="",
        help="GPU info to display in header (e.g., 'H200 141GB')",
    )

    args = parser.parse_args()

    STORAGE_URL = args.storage
    STUDY_NAME = args.study
    REFRESH_INTERVAL = args.refresh
    TARGET_TRIALS = args.target_trials
    TOTAL_PROMPTS = args.total_prompts
    MODEL_NAME = args.model
    GPU_INFO = args.gpu

    # Initialize monitor
    monitor = StudyMonitor(STORAGE_URL, STUDY_NAME)

    logger.info("Bruno Monitor - Starting...")
    logger.info(f"Storage: {STORAGE_URL}")
    logger.info(f"Study: {STUDY_NAME}")
    logger.info(f"Refresh interval: {REFRESH_INTERVAL}s")
    logger.info(f"Target trials: {TARGET_TRIALS}")
    logger.info(f"Total prompts: {TOTAL_PROMPTS}")
    if MODEL_NAME:
        logger.info(f"Model: {MODEL_NAME}")
    if GPU_INFO:
        logger.info(f"GPU: {GPU_INFO}")

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
