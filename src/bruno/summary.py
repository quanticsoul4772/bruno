"""Abliteration run summary reporting."""

from dataclasses import dataclass, field
from datetime import timedelta

from .feature_tracker import feature_tracker
from .utils import print as rich_print


@dataclass
class AbliterationSummary:
    """Complete abliteration run summary."""

    # Execution metrics
    total_trials: int
    successful_trials: int
    failed_trials: int
    total_duration_seconds: float

    # Model info
    model_name: str
    model_size: str
    final_model_path: str

    # Quality metrics
    best_trial_number: int
    best_kl_divergence: float
    best_refusal_count: int
    initial_refusals: int

    # Optimization parameters
    best_parameters: dict

    # Error tracking
    total_errors_suppressed: int = 0
    error_categories: dict[str, int] = field(default_factory=dict)

    # Advanced features (populated from feature_tracker)
    features_active: list[str] = field(default_factory=list)
    features_failed: list[dict] = field(default_factory=list)

    def print_report(self):
        """Print formatted summary report."""
        duration = timedelta(seconds=int(self.total_duration_seconds))

        rich_print("\n" + "=" * 80)
        rich_print("[bold cyan]ABLITERATION SUMMARY REPORT[/]")
        rich_print("=" * 80)

        # Model Information
        rich_print("\n[bold]Model Information:[/]")
        rich_print(f"  Model: {self.model_name}")
        rich_print(f"  Size: {self.model_size}")
        rich_print(f"  Output: {self.final_model_path}")

        # Execution Statistics
        rich_print("\n[bold]Execution Statistics:[/]")
        rich_print(f"  Total Trials: {self.total_trials}")
        rich_print(
            f"  Successful: {self.successful_trials} "
            f"({self.successful_trials / self.total_trials * 100:.1f}%)"
        )
        if self.failed_trials > 0:
            rich_print(f"  [yellow]Failed: {self.failed_trials}[/]")
        rich_print(f"  Duration: {duration}")
        rich_print(
            f"  Avg per trial: {self.total_duration_seconds / self.total_trials:.1f}s"
        )

        # Quality Results
        rich_print("\n[bold]Quality Results:[/]")
        rich_print(f"  Best Trial: #{self.best_trial_number}")
        rich_print(f"  KL Divergence: {self.best_kl_divergence:.4f}")

        refusal_reduction = (
            (self.initial_refusals - self.best_refusal_count)
            / self.initial_refusals
            * 100
        )
        rich_print(
            f"  Refusals: {self.best_refusal_count}/{self.initial_refusals} "
            f"({refusal_reduction:.1f}% reduction)"
        )

        if self.best_kl_divergence > 2.0:
            rich_print(
                "[yellow]  Warning: High KL divergence (>2.0) may indicate capability damage[/]"
            )

        # Best Parameters
        rich_print("\n[bold]Best Parameters:[/]")
        for param, value in self.best_parameters.items():
            if isinstance(value, float):
                rich_print(f"  {param}: {value:.4f}")
            else:
                rich_print(f"  {param}: {value}")

        # Advanced Features Status
        feature_summary = feature_tracker.get_summary()

        if feature_summary["active"]:
            rich_print("\n[bold green]Active Advanced Features:[/]")
            for feature_name in feature_summary["active"]:
                feature = feature_tracker.features[feature_name]
                meta_str = ""
                if feature.metadata:
                    meta_str = f" ({', '.join(f'{k}={v}' for k, v in feature.metadata.items())})"
                rich_print(f"  - {feature.display_name}{meta_str}")

        if feature_summary["failed"]:
            rich_print("\n[bold yellow]Failed Features (used fallback):[/]")
            for failed in feature_summary["failed"]:
                feature = feature_tracker.features[failed["name"]]
                rich_print(f"  - {feature.display_name}")
                rich_print(f"     Reason: {failed['reason']}")
                rich_print(f"     Fallback: {failed.get('fallback', 'None')}")

        # Error Summary
        if self.total_errors_suppressed > 0:
            rich_print("\n[bold]Suppressed Errors:[/]")
            rich_print(f"  Total: {self.total_errors_suppressed}")
            if self.error_categories:
                rich_print("  By Category:")
                for category, count in sorted(
                    self.error_categories.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    rich_print(f"    • {category}: {count}")

        # Recommendations
        rich_print("\n[bold]Recommendations:[/]")

        if refusal_reduction < 80:
            rich_print("  • [yellow]Low refusal reduction. Consider:[/]")
            rich_print(f"    - Running more trials (current: {self.total_trials})")
            rich_print("    - Adjusting harmful_prompts dataset")
            rich_print("    - Enabling more advanced features")

        if self.best_kl_divergence > 2.0:
            rich_print(
                "  • [yellow]High KL divergence. Validate model capabilities:[/]"
            )
            rich_print("    - Run MMLU validation: --run-mmlu-validation")
            rich_print("    - Test on sample prompts before deployment")
            rich_print("    - Consider enabling sacred direction preservation")

        if feature_summary["failed"]:
            rich_print("  • [yellow]Some advanced features failed. Consider:[/]")
            for failed in feature_summary["failed"]:
                if "imbalance" in failed.get("reason", "").lower():
                    rich_print("    - Increase dataset size for balanced sampling")
                elif "accuracy" in failed.get("reason", "").lower():
                    rich_print("    - Improve probe training data quality")

        rich_print("\n" + "=" * 80)
