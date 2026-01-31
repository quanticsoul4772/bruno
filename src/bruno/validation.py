# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Validation framework for measuring abliteration effectiveness.

This module provides tools to:
- Establish baseline metrics before abliteration
- Measure improvement after abliteration
- Run MMLU subset evaluation for capability testing
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset

from .error_tracker import record_suppressed_error
from .exceptions import ValidationFileError
from .logging import get_logger
from .utils import print

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .config import Settings
    from .evaluator import Evaluator
    from .model import Model

# Default MMLU categories for lightweight evaluation
# These cover diverse reasoning domains without being too slow
DEFAULT_MMLU_CATEGORIES = [
    "abstract_algebra",  # Mathematical reasoning
    "high_school_physics",  # Scientific reasoning
    "professional_law",  # Legal/textual reasoning
]

# MMLU answer mapping
MMLU_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


@dataclass
class MMLUResult:
    """Results from MMLU evaluation on a single category."""

    category: str
    correct: int
    total: int
    accuracy: float
    examples: list[dict] = field(default_factory=list)  # Sample predictions


@dataclass
class ValidationMetrics:
    """Metrics collected during validation."""

    refusal_count: int
    refusal_rate: float
    total_bad_prompts: int
    kl_divergence: float
    mmlu_scores: dict[str, float] = field(default_factory=dict)  # category -> accuracy
    mmlu_average: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationReport:
    """Complete validation report comparing baseline to post-abliteration."""

    model_name: str
    baseline: ValidationMetrics
    post_abliteration: ValidationMetrics | None = None

    # Computed improvements (populated when post_abliteration is set)
    refusal_reduction: float = 0.0  # Absolute reduction in refusal rate
    refusal_reduction_pct: float = 0.0  # Percentage improvement
    kl_divergence_increase: float = 0.0
    mmlu_change: float = 0.0  # Change in MMLU average (negative = degradation)
    capability_preserved: bool = True  # True if KL < 1.0 and MMLU drop < 5%

    def compute_improvements(self) -> None:
        """Compute improvement metrics after post_abliteration is set."""
        if self.post_abliteration is None:
            return

        self.refusal_reduction = (
            self.baseline.refusal_rate - self.post_abliteration.refusal_rate
        )
        if self.baseline.refusal_rate > 0:
            self.refusal_reduction_pct = (
                self.refusal_reduction / self.baseline.refusal_rate
            ) * 100
        else:
            self.refusal_reduction_pct = 0.0

        self.kl_divergence_increase = self.post_abliteration.kl_divergence

        if self.baseline.mmlu_average > 0 and self.post_abliteration.mmlu_average > 0:
            self.mmlu_change = (
                self.post_abliteration.mmlu_average - self.baseline.mmlu_average
            )
        else:
            self.mmlu_change = 0.0

        # Capability preserved if KL divergence is reasonable and MMLU didn't drop much
        self.capability_preserved = (
            self.kl_divergence_increase < 1.0 and self.mmlu_change > -0.05
        )

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "baseline": asdict(self.baseline),
            "post_abliteration": asdict(self.post_abliteration)
            if self.post_abliteration
            else None,
            "improvements": {
                "refusal_reduction": self.refusal_reduction,
                "refusal_reduction_pct": self.refusal_reduction_pct,
                "kl_divergence_increase": self.kl_divergence_increase,
                "mmlu_change": self.mmlu_change,
                "capability_preserved": self.capability_preserved,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file with atomic write and error handling."""
        path = Path(path)

        # Validate parent directory
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            print("[red]Permission Denied: Cannot create directory[/]")
            print(f"[yellow]Path: {path.parent}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check directory permissions")
            print("  2. Try a different output directory")
            print("  3. Run with appropriate permissions")
            raise ValidationFileError(
                f"Permission denied creating directory: {path.parent}"
            ) from e
        except OSError as e:
            error_msg = str(e).lower()
            if "disk" in error_msg or "space" in error_msg or "storage" in error_msg:
                print("[red]Insufficient Disk Space[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Free up disk space")
                print("  2. Choose different output directory")
                raise ValidationFileError(
                    f"Insufficient disk space to create directory: {path.parent}"
                ) from e
            raise ValidationFileError(
                f"Failed to create directory: {path.parent}"
            ) from e

        # Check disk space (need ~1MB for JSON report)
        try:
            stat = os.statvfs(path.parent) if hasattr(os, "statvfs") else None
            if stat and (stat.f_bavail * stat.f_frsize) < 1_000_000:  # < 1MB
                print("[red]Insufficient Disk Space (< 1MB available)[/]")
                print("[yellow]Free up space before saving report[/]")
                raise ValidationFileError(
                    f"Insufficient disk space to save report: {path}"
                )
        except AttributeError as e:
            # statvfs not available on Windows - skip check
            record_suppressed_error(
                error=e,
                context="check_disk_space",
                module="validation",
                severity="debug",
                details={
                    "path": str(path),
                    "reason": "statvfs not available (Windows)",
                },
            )

        # Atomic write: write to temp file, then rename
        # This ensures we never have a partial/corrupted file
        temp_fd = None
        temp_path = None
        try:
            # Create temp file in same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )

            # Write JSON to temp file
            with os.fdopen(temp_fd, "w") as f:
                temp_fd = None  # fdopen took ownership
                json.dump(self.to_dict(), f, indent=2)

            # Atomic rename (on POSIX) or best-effort rename (on Windows)
            Path(temp_path).replace(path)

        except PermissionError as e:
            print("[red]Permission Denied: Cannot write file[/]")
            print(f"[yellow]Path: {path}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check file permissions")
            print("  2. Close any programs using this file")
            print("  3. Try a different output path")
            raise ValidationFileError(f"Permission denied writing file: {path}") from e
        except OSError as e:
            error_msg = str(e).lower()
            if "disk" in error_msg or "space" in error_msg or "storage" in error_msg:
                print("[red]Disk Full: Cannot write file[/]")
                print("[yellow]Free up disk space and try again[/]")
                raise ValidationFileError(
                    f"Disk full while writing file: {path}"
                ) from e
            raise ValidationFileError(f"Failed to write file: {path}") from e
        except (TypeError, ValueError) as e:
            # JSON serialization error
            print("[red]Serialization Error: Cannot convert report to JSON[/]")
            print(f"[yellow]Error: {e}[/]")
            raise ValidationFileError(
                f"Failed to serialize validation report to JSON: {e}"
            ) from e
        finally:
            # Clean up temp file if something went wrong
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError as cleanup_error:
                    record_suppressed_error(
                        error=cleanup_error,
                        context="save_cleanup_temp",
                        module="validation",
                        severity="debug",
                        details={"temp_path": str(temp_path)},
                    )
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except OSError as cleanup_error:
                    record_suppressed_error(
                        error=cleanup_error,
                        context="save_cleanup_temp",
                        module="validation",
                        severity="debug",
                        details={"temp_path": str(temp_path)},
                    )

    @classmethod
    def load(cls, path: str | Path) -> "ValidationReport":
        """Load report from JSON file with validation and error handling."""
        path = Path(path)

        # Check file exists
        if not path.exists():
            print(f"[red]File Not Found: {path}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check file path spelling")
            print("  2. Verify file hasn't been moved or deleted")
            print("  3. Check current working directory")
            raise ValidationFileError(f"Validation report not found: {path}")

        # Check file is readable
        if not os.access(path, os.R_OK):
            print("[red]Permission Denied: Cannot read file[/]")
            print(f"[yellow]Path: {path}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check file permissions")
            print("  2. Run with appropriate permissions")
            raise ValidationFileError(f"Permission denied reading file: {path}")

        # Load and parse JSON
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print("[red]Corrupted File: Invalid JSON[/]")
            print(f"[yellow]Path: {path}[/]")
            print(f"[yellow]Error at line {e.lineno}, column {e.colno}:[/]")
            print(f"  {e.msg}")
            print("[yellow]Solutions:[/]")
            print("  1. File may be corrupted - restore from backup")
            print("  2. Regenerate validation report")
            print("  3. Check file wasn't manually edited")
            raise ValidationFileError(
                f"Corrupted validation report - invalid JSON at line {e.lineno}, "
                f"column {e.colno}: {e.msg}"
            ) from e
        except UnicodeDecodeError as e:
            print("[red]Encoding Error: Cannot decode file[/]")
            print(f"[yellow]Path: {path}[/]")
            print("[yellow]File may be corrupted or in wrong format[/]")
            raise ValidationFileError(
                f"Failed to decode validation report (encoding error): {path}"
            ) from e

        # Validate required fields
        required_fields = ["model_name", "baseline", "post_abliteration"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print("[red]Invalid Report: Missing required fields[/]")
            print(f"[yellow]Path: {path}[/]")
            print(f"[yellow]Missing fields: {missing_fields}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. File may be from old version - regenerate report")
            print("  2. Restore from backup if available")
            raise ValidationFileError(
                f"Invalid validation report - missing fields: {missing_fields}"
            )

        # Parse validation metrics
        try:
            baseline = ValidationMetrics(**data["baseline"])
        except (TypeError, KeyError) as e:
            print("[red]Invalid Baseline Metrics[/]")
            print(f"[yellow]Error: {e}[/]")
            print("[yellow]Regenerate validation report[/]")
            raise ValidationFileError(
                f"Invalid baseline metrics in validation report: {e}"
            ) from e

        try:
            post = (
                ValidationMetrics(**data["post_abliteration"])
                if data["post_abliteration"]
                else None
            )
        except (TypeError, KeyError) as e:
            print("[red]Invalid Post-Abliteration Metrics[/]")
            print(f"[yellow]Error: {e}[/]")
            print("[yellow]Regenerate validation report[/]")
            raise ValidationFileError(
                f"Invalid post-abliteration metrics in validation report: {e}"
            ) from e

        # Create report
        try:
            report = cls(
                model_name=data["model_name"],
                baseline=baseline,
                post_abliteration=post,
            )
            if post:
                report.compute_improvements()
            return report
        except Exception as e:
            print("[red]Failed to Create Validation Report[/]")
            print(f"[yellow]Error: {e}[/]")
            raise ValidationFileError(
                f"Failed to create validation report from file: {e}"
            ) from e


class MMLUEvaluator:
    """Lightweight MMLU evaluator for capability testing.

    Uses a subset of MMLU categories to quickly assess model capabilities
    without running the full benchmark.
    """

    def __init__(
        self,
        model: "Model",
        categories: list[str] | None = None,
        samples_per_category: int = 20,
        n_few_shot: int = 3,
    ):
        self.model = model
        self.categories = categories or DEFAULT_MMLU_CATEGORIES
        self.samples_per_category = samples_per_category
        self.n_few_shot = n_few_shot

        # Cache loaded datasets
        self._datasets: dict[str, list[dict]] = {}

    def _load_category(self, category: str) -> list[dict]:
        """Load MMLU dataset for a specific category."""
        if category in self._datasets:
            return self._datasets[category]

        try:
            # Load from cais/mmlu with category as config name
            dataset = load_dataset("cais/mmlu", category, split="test")

            # Convert to list of dicts with standardized format
            examples = []
            for item in dataset:
                # Handle both integer (0-3) and string ('A'-'D') answer formats
                answer = item["answer"]
                if isinstance(answer, int):
                    answer = MMLU_INDEX_TO_LETTER[answer]
                
                # The cais/mmlu dataset has 'choices' as a list, not separate A/B/C/D columns
                examples.append(
                    {
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": answer,
                    }
                )

            self._datasets[category] = examples
            return examples

        except Exception as e:
            print(f"[yellow]Warning: Failed to load MMLU category {category}: {e}[/]")
            return []

    def _format_question(self, question: str, choices: list[str]) -> str:
        """Format a single MMLU question."""
        formatted = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            letter = MMLU_INDEX_TO_LETTER[i]
            formatted += f"{letter}. {choice}\n"
        formatted += "Answer:"
        return formatted

    def _format_few_shot_prompt(self, examples: list[dict], test_question: dict) -> str:
        """Format a few-shot prompt with examples."""
        prompt_parts = [
            "The following are multiple choice questions. "
            "Select the correct answer from the options provided.\n\n"
        ]

        # Add few-shot examples
        for ex in examples[: self.n_few_shot]:
            q = self._format_question(ex["question"], ex["choices"])
            prompt_parts.append(f"{q} {ex['answer']}\n\n")

        # Add test question
        test_q = self._format_question(
            test_question["question"], test_question["choices"]
        )
        prompt_parts.append(test_q)

        return "".join(prompt_parts)

    def _extract_answer(self, response: str) -> str | None:
        """Extract the answer letter from model response."""
        response = response.strip().upper()

        # Check if response starts with a letter
        if response and response[0] in "ABCD":
            return response[0]

        # If response is very short (just the answer), return None if not found
        if len(response) <= 3:
            logger.debug(
                "MMLU answer extraction failed - no valid answer found",
                response_preview=response[:50] if response else "(empty)",
            )
            return None

        logger.debug(
            "MMLU answer extraction failed - response too long without clear answer",
            response_length=len(response),
            response_preview=response[:50],
        )
        return None

    def evaluate_category(self, category: str) -> MMLUResult:
        """Evaluate model on a single MMLU category."""
        examples = self._load_category(category)

        if not examples:
            return MMLUResult(
                category=category, correct=0, total=0, accuracy=0.0, examples=[]
            )

        # Use first n_few_shot examples for few-shot, rest for testing
        few_shot_examples = examples[: self.n_few_shot]
        test_examples = examples[self.n_few_shot :][: self.samples_per_category]

        correct = 0
        total = len(test_examples)
        sample_results = []

        # Process in batches using model's batch processing
        prompts = [
            self._format_few_shot_prompt(few_shot_examples, ex) for ex in test_examples
        ]

        # Get responses (limit tokens since we only need the answer letter)
        responses = self.model.get_responses_batched(prompts, max_tokens=5)

        for ex, response in zip(test_examples, responses):
            predicted = self._extract_answer(response)
            expected = ex["answer"]
            is_correct = predicted == expected

            if is_correct:
                correct += 1

            # Store first few examples for debugging
            if len(sample_results) < 3:
                sample_results.append(
                    {
                        "question": ex["question"][:100] + "..."
                        if len(ex["question"]) > 100
                        else ex["question"],
                        "expected": expected,
                        "predicted": predicted,
                        "correct": is_correct,
                    }
                )

        accuracy = correct / total if total > 0 else 0.0

        return MMLUResult(
            category=category,
            correct=correct,
            total=total,
            accuracy=accuracy,
            examples=sample_results,
        )

    def evaluate(self) -> dict[str, float]:
        """Evaluate model on all configured MMLU categories.

        Returns:
            Dictionary mapping category names to accuracy scores.
        """
        results = {}

        for category in self.categories:
            print(f"    * Evaluating MMLU category: [bold]{category}[/]...")
            result = self.evaluate_category(category)
            results[category] = result.accuracy
            print(
                f"      * Accuracy: [bold]{result.accuracy:.1%}[/] "
                f"({result.correct}/{result.total})"
            )

        return results


class AbliterationValidator:
    """Validates abliteration effectiveness with comprehensive metrics.

    This class provides a framework for:
    1. Establishing baseline metrics before abliteration
    2. Measuring improvement after abliteration
    3. Running MMLU subset evaluation for capability testing
    4. Generating validation reports

    Usage:
        validator = AbliterationValidator(settings, model, evaluator)
        validator.establish_baseline()

        # ... perform abliteration ...

        validator.measure_post_abliteration()
        report = validator.get_report()
        report.save("validation_report.json")
    """

    def __init__(
        self,
        settings: "Settings",
        model: "Model",
        evaluator: "Evaluator",
    ):
        self.settings = settings
        self.model = model
        self.evaluator = evaluator

        # Initialize MMLU evaluator if enabled
        self.mmlu_evaluator: MMLUEvaluator | None = None
        if settings.enable_validation and settings.run_mmlu_validation:
            self.mmlu_evaluator = MMLUEvaluator(
                model=model,
                categories=settings.mmlu_categories,
                samples_per_category=settings.mmlu_samples_per_category,
                n_few_shot=settings.mmlu_few_shot,
            )

        self.baseline: ValidationMetrics | None = None
        self.post_abliteration: ValidationMetrics | None = None

    def _collect_metrics(
        self, include_mmlu: bool = True, is_baseline: bool = False
    ) -> ValidationMetrics:
        """Collect current model metrics."""
        # Get refusal metrics from evaluator
        refusals = self.evaluator.count_refusals()
        total = len(self.evaluator.bad_prompts)
        refusal_rate = refusals / total if total > 0 else 0.0

        # KL divergence (0 for baseline by definition)
        if is_baseline:
            kl_divergence = 0.0
        else:
            kl_divergence = self.evaluator._compute_kl_divergence()

        # MMLU evaluation
        mmlu_scores = {}
        mmlu_average = 0.0
        if include_mmlu and self.mmlu_evaluator is not None:
            print("  * Running MMLU capability evaluation...")
            mmlu_scores = self.mmlu_evaluator.evaluate()
            if mmlu_scores:
                mmlu_average = sum(mmlu_scores.values()) / len(mmlu_scores)
                print(f"  * MMLU average accuracy: [bold]{mmlu_average:.1%}[/]")

        return ValidationMetrics(
            refusal_count=refusals,
            refusal_rate=refusal_rate,
            total_bad_prompts=total,
            kl_divergence=kl_divergence,
            mmlu_scores=mmlu_scores,
            mmlu_average=mmlu_average,
        )

    def establish_baseline(self) -> ValidationMetrics:
        """Establish baseline metrics before abliteration.

        This should be called after the model is loaded but before any
        abliteration is performed.

        Returns:
            ValidationMetrics containing baseline measurements.
        """
        print()
        print("[cyan]Establishing validation baseline...[/]")

        self.baseline = self._collect_metrics(
            include_mmlu=self.settings.run_mmlu_validation, is_baseline=True
        )

        print(
            f"  * Baseline refusal rate: [bold]{self.baseline.refusal_rate:.1%}[/] "
            f"({self.baseline.refusal_count}/{self.baseline.total_bad_prompts})"
        )

        return self.baseline

    def measure_post_abliteration(self) -> ValidationMetrics:
        """Measure metrics after abliteration.

        This should be called after abliteration is complete.

        Returns:
            ValidationMetrics containing post-abliteration measurements.
        """
        print()
        print("[cyan]Measuring post-abliteration metrics...[/]")

        self.post_abliteration = self._collect_metrics(
            include_mmlu=self.settings.run_mmlu_validation, is_baseline=False
        )

        print(
            f"  * Post-abliteration refusal rate: "
            f"[bold]{self.post_abliteration.refusal_rate:.1%}[/] "
            f"({self.post_abliteration.refusal_count}/{self.post_abliteration.total_bad_prompts})"
        )
        print(f"  * KL divergence: [bold]{self.post_abliteration.kl_divergence:.2f}[/]")

        return self.post_abliteration

    def get_report(self) -> ValidationReport:
        """Generate a validation report.

        Returns:
            ValidationReport with baseline and post-abliteration comparison.

        Raises:
            ValueError: If baseline has not been established.
        """
        if self.baseline is None:
            raise ValueError(
                "Baseline not established. Call establish_baseline() first."
            )

        report = ValidationReport(
            model_name=self.settings.model,
            baseline=self.baseline,
            post_abliteration=self.post_abliteration,
        )

        if self.post_abliteration is not None:
            report.compute_improvements()

        return report

    def print_summary(self) -> None:
        """Print a summary of the validation results."""
        report = self.get_report()

        print()
        print("[bold cyan]═══ Validation Summary ═══[/]")
        print()

        # Baseline metrics
        print("[bold]Baseline:[/]")
        print(
            f"  Refusal rate: {report.baseline.refusal_rate:.1%} "
            f"({report.baseline.refusal_count}/{report.baseline.total_bad_prompts})"
        )
        if report.baseline.mmlu_average > 0:
            print(f"  MMLU average: {report.baseline.mmlu_average:.1%}")
            for cat, acc in report.baseline.mmlu_scores.items():
                print(f"    - {cat}: {acc:.1%}")

        # Post-abliteration metrics
        if report.post_abliteration is not None:
            print()
            print("[bold]Post-Abliteration:[/]")
            print(
                f"  Refusal rate: {report.post_abliteration.refusal_rate:.1%} "
                f"({report.post_abliteration.refusal_count}/{report.post_abliteration.total_bad_prompts})"
            )
            print(f"  KL divergence: {report.post_abliteration.kl_divergence:.2f}")
            if report.post_abliteration.mmlu_average > 0:
                print(f"  MMLU average: {report.post_abliteration.mmlu_average:.1%}")
                for cat, acc in report.post_abliteration.mmlu_scores.items():
                    print(f"    - {cat}: {acc:.1%}")

            # Improvements
            print()
            print("[bold]Improvements:[/]")
            color = "green" if report.refusal_reduction > 0 else "red"
            print(
                f"  Refusal reduction: [{color}]{report.refusal_reduction:.1%} "
                f"({report.refusal_reduction_pct:.0f}% improvement)[/]"
            )

            if report.mmlu_change != 0:
                color = "green" if report.mmlu_change >= 0 else "yellow"
                direction = "+" if report.mmlu_change >= 0 else ""
                print(f"  MMLU change: [{color}]{direction}{report.mmlu_change:.1%}[/]")

            # Overall verdict
            print()
            if report.capability_preserved:
                print("[bold green]✓ Capability preserved (KL < 1.0, MMLU stable)[/]")
            else:
                print("[bold yellow]⚠ Potential capability degradation detected[/]")

        print()
        print("[cyan]═══════════════════════════[/]")
