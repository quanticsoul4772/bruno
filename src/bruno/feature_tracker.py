"""Advanced feature lifecycle tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .logging import get_logger

logger = get_logger(__name__)


class FeatureState(Enum):
    """Feature lifecycle states."""

    REQUESTED = "requested"  # User enabled via config
    ATTEMPTING = "attempting"  # Currently trying to activate
    ACTIVE = "active"  # Successfully activated
    FAILED = "failed"  # Attempted but failed
    DISABLED = "disabled"  # Not requested by user
    UNAVAILABLE = "unavailable"  # Not supported for this model


@dataclass
class FeatureStatus:
    """Track individual feature status."""

    name: str
    display_name: str
    state: FeatureState
    failure_reason: Optional[str] = None
    impact: Optional[str] = None
    fallback: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class FeatureTracker:
    """Central registry for advanced feature tracking."""

    def __init__(self):
        self.features: dict[str, FeatureStatus] = {}
        self._init_features()

    def _init_features(self):
        """Initialize all trackable features."""
        feature_definitions = {
            "neural_detection": "Neural Refusal Detection",
            "supervised_probing": "Supervised Probing",
            "ensemble_probe_pca": "Ensemble (Probe + PCA)",
            "activation_calibration": "Activation Calibration",
            "concept_cones": "Concept Cones",
            "caa": "CAA (Contrastive Activation Addition)",
            "circuit_ablation": "Circuit-Level Ablation",
            "warm_start": "Warm-Start Transfer",
            "helpfulness_orthogonalization": "Helpfulness Orthogonalization",
            "moe_targeting": "MoE Expert Targeting",
        }

        for key, display_name in feature_definitions.items():
            self.features[key] = FeatureStatus(
                name=key, display_name=display_name, state=FeatureState.DISABLED
            )

    def request(self, feature: str):
        """Mark feature as requested by user."""
        if feature in self.features:
            self.features[feature].state = FeatureState.REQUESTED
            logger.debug(f"Feature requested: {feature}")

    def attempt(self, feature: str):
        """Mark feature activation attempt."""
        if feature in self.features:
            self.features[feature].state = FeatureState.ATTEMPTING
            logger.debug(f"Attempting to activate: {feature}")

    def activate(self, feature: str, metadata: dict = None):
        """Mark feature as successfully activated."""
        if feature in self.features:
            self.features[feature].state = FeatureState.ACTIVE
            if metadata:
                self.features[feature].metadata = metadata
            logger.info(f"Feature activated: {feature}")

    def fail(self, feature: str, reason: str, impact: str, fallback: str = None):
        """Mark feature as failed with context."""
        if feature in self.features:
            self.features[feature].state = FeatureState.FAILED
            self.features[feature].failure_reason = reason
            self.features[feature].impact = impact
            self.features[feature].fallback = fallback
            logger.warning(f"Feature failed: {feature} - {reason} - Impact: {impact}")

    def mark_unavailable(self, feature: str, reason: str):
        """Mark feature as unavailable for this model."""
        if feature in self.features:
            self.features[feature].state = FeatureState.UNAVAILABLE
            self.features[feature].failure_reason = reason
            logger.info(f"Feature unavailable: {feature} - {reason}")

    def get_summary(self) -> dict:
        """Get summary of all feature states."""
        summary = {
            "requested": [],
            "active": [],
            "failed": [],
            "unavailable": [],
        }

        for feature in self.features.values():
            if feature.state == FeatureState.REQUESTED:
                summary["requested"].append(feature.name)
            elif feature.state == FeatureState.ACTIVE:
                summary["active"].append(feature.name)
            elif feature.state == FeatureState.FAILED:
                summary["failed"].append(
                    {
                        "name": feature.name,
                        "reason": feature.failure_reason,
                        "impact": feature.impact,
                        "fallback": feature.fallback,
                    }
                )
            elif feature.state == FeatureState.UNAVAILABLE:
                summary["unavailable"].append(
                    {
                        "name": feature.name,
                        "reason": feature.failure_reason,
                    }
                )

        return summary

    def print_status(self):
        """Print formatted feature status report."""
        from .utils import print as rich_print

        rich_print("\n[bold]Advanced Features Status:[/]")

        # Active features
        active = [f for f in self.features.values() if f.state == FeatureState.ACTIVE]
        if active:
            rich_print("\n[green]Active Features:[/]")
            for feature in active:
                meta_str = ""
                if feature.metadata:
                    meta_str = f" ({', '.join(f'{k}={v}' for k, v in feature.metadata.items())})"
                rich_print(f"  - {feature.display_name}{meta_str}")

        # Failed features
        failed = [f for f in self.features.values() if f.state == FeatureState.FAILED]
        if failed:
            rich_print("\n[yellow]Failed Features (using fallback):[/]")
            for feature in failed:
                rich_print(f"  - {feature.display_name}")
                rich_print(f"    Reason: {feature.failure_reason}")
                rich_print(f"    Impact: {feature.impact}")
                if feature.fallback:
                    rich_print(f"    Using: {feature.fallback}")

        # Unavailable features
        unavailable = [
            f for f in self.features.values() if f.state == FeatureState.UNAVAILABLE
        ]
        if unavailable:
            rich_print("\n[dim]Unavailable Features:[/]")
            for feature in unavailable:
                rich_print(f"  - {feature.display_name}: {feature.failure_reason}")

        # Disabled features
        disabled = [
            f for f in self.features.values() if f.state == FeatureState.DISABLED
        ]
        if disabled:
            rich_print("\n[dim]Disabled Features:[/]")
            for feature in disabled:
                rich_print(f"  - {feature.display_name}")


# Global feature tracker instance
feature_tracker = FeatureTracker()
