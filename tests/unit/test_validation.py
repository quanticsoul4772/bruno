# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for validation framework."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestValidationMetrics:
    """Test ValidationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating ValidationMetrics with valid values."""
        from bruno.validation import ValidationMetrics

        metrics = ValidationMetrics(
            refusal_count=10,
            refusal_rate=0.5,
            total_bad_prompts=20,
            kl_divergence=0.5,
            mmlu_scores={"abstract_algebra": 0.6},
            mmlu_average=0.6,
        )

        assert metrics.refusal_count == 10
        assert metrics.refusal_rate == 0.5
        assert metrics.kl_divergence == 0.5
        assert metrics.mmlu_average == 0.6

    def test_metrics_default_mmlu(self):
        """Test ValidationMetrics with default MMLU values."""
        from bruno.validation import ValidationMetrics

        metrics = ValidationMetrics(
            refusal_count=5,
            refusal_rate=0.25,
            total_bad_prompts=20,
            kl_divergence=0.0,
        )

        assert metrics.mmlu_scores == {}
        assert metrics.mmlu_average == 0.0


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_report_creation(self):
        """Test creating ValidationReport."""
        from bruno.validation import ValidationMetrics, ValidationReport

        baseline = ValidationMetrics(
            refusal_count=20,
            refusal_rate=0.8,
            total_bad_prompts=25,
            kl_divergence=0.0,
            mmlu_average=0.7,
        )

        report = ValidationReport(
            model_name="test-model",
            baseline=baseline,
        )

        assert report.model_name == "test-model"
        assert report.baseline.refusal_rate == 0.8
        assert report.post_abliteration is None

    def test_report_compute_improvements(self):
        """Test computing improvement metrics."""
        from bruno.validation import ValidationMetrics, ValidationReport

        baseline = ValidationMetrics(
            refusal_count=20,
            refusal_rate=0.8,
            total_bad_prompts=25,
            kl_divergence=0.0,
            mmlu_average=0.7,
        )

        post = ValidationMetrics(
            refusal_count=5,
            refusal_rate=0.2,
            total_bad_prompts=25,
            kl_divergence=0.5,
            mmlu_average=0.68,
        )

        report = ValidationReport(
            model_name="test-model",
            baseline=baseline,
            post_abliteration=post,
        )
        report.compute_improvements()

        assert report.refusal_reduction == pytest.approx(0.6)  # 0.8 - 0.2
        assert report.refusal_reduction_pct == pytest.approx(75.0)  # 75% improvement
        assert report.kl_divergence_increase == 0.5
        assert report.mmlu_change == pytest.approx(-0.02)  # 0.68 - 0.70
        assert report.capability_preserved is True  # KL < 1.0, MMLU drop < 5%

    def test_report_capability_not_preserved_high_kl(self):
        """Test capability_preserved is False when KL is too high."""
        from bruno.validation import ValidationMetrics, ValidationReport

        baseline = ValidationMetrics(
            refusal_count=20,
            refusal_rate=0.8,
            total_bad_prompts=25,
            kl_divergence=0.0,
            mmlu_average=0.7,
        )

        post = ValidationMetrics(
            refusal_count=2,
            refusal_rate=0.08,
            total_bad_prompts=25,
            kl_divergence=1.5,  # Too high
            mmlu_average=0.7,
        )

        report = ValidationReport(
            model_name="test-model",
            baseline=baseline,
            post_abliteration=post,
        )
        report.compute_improvements()

        assert report.capability_preserved is False

    def test_report_capability_not_preserved_mmlu_drop(self):
        """Test capability_preserved is False when MMLU drops too much."""
        from bruno.validation import ValidationMetrics, ValidationReport

        baseline = ValidationMetrics(
            refusal_count=20,
            refusal_rate=0.8,
            total_bad_prompts=25,
            kl_divergence=0.0,
            mmlu_average=0.7,
        )

        post = ValidationMetrics(
            refusal_count=2,
            refusal_rate=0.08,
            total_bad_prompts=25,
            kl_divergence=0.3,
            mmlu_average=0.5,  # 20% drop
        )

        report = ValidationReport(
            model_name="test-model",
            baseline=baseline,
            post_abliteration=post,
        )
        report.compute_improvements()

        assert report.capability_preserved is False

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        from bruno.validation import ValidationMetrics, ValidationReport

        baseline = ValidationMetrics(
            refusal_count=10,
            refusal_rate=0.5,
            total_bad_prompts=20,
            kl_divergence=0.0,
        )

        report = ValidationReport(
            model_name="test-model",
            baseline=baseline,
        )

        d = report.to_dict()

        assert d["model_name"] == "test-model"
        assert d["baseline"]["refusal_count"] == 10
        assert d["post_abliteration"] is None

    def test_report_save_load(self):
        """Test saving and loading report from JSON."""
        from bruno.validation import ValidationMetrics, ValidationReport

        baseline = ValidationMetrics(
            refusal_count=15,
            refusal_rate=0.6,
            total_bad_prompts=25,
            kl_divergence=0.0,
            mmlu_scores={"algebra": 0.7},
            mmlu_average=0.7,
        )

        post = ValidationMetrics(
            refusal_count=3,
            refusal_rate=0.12,
            total_bad_prompts=25,
            kl_divergence=0.4,
            mmlu_scores={"algebra": 0.68},
            mmlu_average=0.68,
        )

        report = ValidationReport(
            model_name="test-model",
            baseline=baseline,
            post_abliteration=post,
        )
        report.compute_improvements()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["model_name"] == "test-model"

            # Load and verify
            loaded = ValidationReport.load(path)
            assert loaded.model_name == "test-model"
            assert loaded.baseline.refusal_count == 15
            assert loaded.post_abliteration is not None
            assert loaded.post_abliteration.kl_divergence == 0.4


class TestMMLUCategoryValidation:
    """Test MMLU category name validation.

    These tests ensure all configured MMLU category names are valid.
    This prevents bugs like using 'world_history' instead of 'high_school_world_history'.
    """

    # Complete list of valid MMLU categories from cais/mmlu dataset
    VALID_MMLU_CATEGORIES = {
        "abstract_algebra",
        "all",
        "anatomy",
        "astronomy",
        "auxiliary_train",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    }

    def test_default_mmlu_categories_are_valid(self):
        """Test that all default MMLU categories in Settings are valid.

        This test prevents bugs like using 'world_history' instead of
        'high_school_world_history' (which was a real bug fixed in commit 10e7d11).
        """
        from bruno.config import Settings

        settings = Settings(model="test-model")

        invalid_categories = []
        for category in settings.mmlu_categories:
            if category not in self.VALID_MMLU_CATEGORIES:
                invalid_categories.append(category)

        assert not invalid_categories, (
            f"Invalid MMLU categories found in Settings.mmlu_categories: {invalid_categories}. "
            f"Valid categories are: {sorted(self.VALID_MMLU_CATEGORIES)}"
        )

    def test_default_mmlu_categories_module_constant_is_valid(self):
        """Test that DEFAULT_MMLU_CATEGORIES constant contains valid categories."""
        from bruno.validation import DEFAULT_MMLU_CATEGORIES

        invalid_categories = []
        for category in DEFAULT_MMLU_CATEGORIES:
            if category not in self.VALID_MMLU_CATEGORIES:
                invalid_categories.append(category)

        assert not invalid_categories, (
            f"Invalid MMLU categories found in DEFAULT_MMLU_CATEGORIES: {invalid_categories}. "
            f"Valid categories are: {sorted(self.VALID_MMLU_CATEGORIES)}"
        )

    def test_known_invalid_category_would_fail(self):
        """Test that the validation would catch the old 'world_history' bug."""
        # This is a regression test - 'world_history' was incorrectly used
        # instead of 'high_school_world_history'
        assert "world_history" not in self.VALID_MMLU_CATEGORIES
        assert "high_school_world_history" in self.VALID_MMLU_CATEGORIES

    def test_common_typos_are_invalid(self):
        """Test that common category name typos are not in valid set."""
        # These are plausible typos that should NOT be valid
        invalid_typos = [
            "abstract-algebra",  # hyphen instead of underscore
            "abstractalgebra",  # no separator
            "high_school_history",  # too generic
            "world_history",  # missing 'high_school_' prefix
            "us_history",  # missing 'high_school_' prefix
            "european_history",  # missing 'high_school_' prefix
            "computer_science",  # ambiguous (college vs high_school)
            "biology",  # ambiguous (college vs high_school)
            "chemistry",  # ambiguous (college vs high_school)
            "physics",  # ambiguous (college vs high_school)
            "mathematics",  # ambiguous (college vs high_school vs elementary)
        ]

        for typo in invalid_typos:
            assert typo not in self.VALID_MMLU_CATEGORIES, (
                f"'{typo}' should not be a valid MMLU category"
            )


class TestMMLUResult:
    """Test MMLUResult dataclass."""

    def test_mmlu_result_creation(self):
        """Test creating MMLUResult."""
        from bruno.validation import MMLUResult

        result = MMLUResult(
            category="abstract_algebra",
            correct=15,
            total=20,
            accuracy=0.75,
            examples=[{"question": "Q1", "expected": "A", "predicted": "A"}],
        )

        assert result.category == "abstract_algebra"
        assert result.accuracy == 0.75
        assert len(result.examples) == 1


class TestMMLUEvaluator:
    """Test MMLUEvaluator class."""

    def test_format_question(self):
        """Test question formatting."""
        from bruno.validation import MMLUEvaluator

        evaluator = MMLUEvaluator(model=MagicMock())

        formatted = evaluator._format_question("What is 2+2?", ["3", "4", "5", "6"])

        assert "Question: What is 2+2?" in formatted
        assert "A. 3" in formatted
        assert "B. 4" in formatted
        assert "C. 5" in formatted
        assert "D. 6" in formatted
        assert "Answer:" in formatted

    def test_load_category_parses_choices_list(self):
        """Test that _load_category correctly parses the 'choices' list format.
        
        The cais/mmlu dataset uses a 'choices' list, NOT separate A/B/C/D columns.
        This was a bug fixed after discovering the dataset structure:
        {'question': '...', 'choices': ['opt1', 'opt2', 'opt3', 'opt4'], 'answer': 1}
        """
        from bruno.validation import MMLUEvaluator
        
        evaluator = MMLUEvaluator(model=MagicMock())
        
        # Mock the dataset loading with the actual cais/mmlu format
        mock_dataset = [
            {
                "question": "Find the degree for the given field extension",
                "subject": "abstract_algebra",
                "choices": ["0", "4", "2", "6"],  # This is how cais/mmlu stores choices
                "answer": 1,  # Integer index (0-3)
            },
            {
                "question": "What is the order of the group?",
                "subject": "abstract_algebra", 
                "choices": ["1", "2", "4", "8"],
                "answer": 2,
            },
        ]
        
        with patch("heretic.validation.load_dataset", return_value=mock_dataset):
            with patch("heretic.validation.print"):
                examples = evaluator._load_category("abstract_algebra")
        
        assert len(examples) == 2
        # Verify choices are parsed correctly from the 'choices' list
        assert examples[0]["choices"] == ["0", "4", "2", "6"]
        assert examples[1]["choices"] == ["1", "2", "4", "8"]
        # Verify answer is converted from int to letter
        assert examples[0]["answer"] == "B"  # index 1 -> B
        assert examples[1]["answer"] == "C"  # index 2 -> C

    def test_extract_answer_single_letter(self):
        """Test extracting answer when response is a single letter."""
        from bruno.validation import MMLUEvaluator

        evaluator = MMLUEvaluator(model=MagicMock())

        assert evaluator._extract_answer("A") == "A"
        assert evaluator._extract_answer("b") == "B"
        assert evaluator._extract_answer(" C ") == "C"
        assert evaluator._extract_answer("D.") == "D"

    def test_extract_answer_with_text(self):
        """Test extracting answer from longer response."""
        from bruno.validation import MMLUEvaluator

        evaluator = MMLUEvaluator(model=MagicMock())

        # These start with the answer letter
        assert evaluator._extract_answer("B is correct") == "B"
        assert evaluator._extract_answer("A. The first option") == "A"

    def test_extract_answer_invalid(self):
        """Test extracting answer from invalid response."""
        from bruno.validation import MMLUEvaluator

        evaluator = MMLUEvaluator(model=MagicMock())

        assert evaluator._extract_answer("") is None
        assert evaluator._extract_answer("XY") is None  # Short but no valid letter

    def test_evaluate_category_with_mocked_data(self):
        """Test evaluating a category with mocked dataset.
        
        Uses the standardized format that _load_category produces:
        {'question': '...', 'choices': [...], 'answer': 'A'/'B'/'C'/'D'}
        """
        from bruno.validation import MMLUEvaluator

        mock_model = MagicMock()
        # Return correct answers for first 2, wrong for last 2
        mock_model.get_responses_batched.return_value = ["A", "B", "C", "D"]

        evaluator = MMLUEvaluator(
            model=mock_model,
            categories=["test_category"],
            samples_per_category=4,
            n_few_shot=2,
        )

        # Mock the dataset in the standardized format (after _load_category processing)
        evaluator._datasets["test_category"] = [
            # Few-shot examples (first 2)
            {"question": "Q1", "choices": ["a", "b", "c", "d"], "answer": "A"},
            {"question": "Q2", "choices": ["a", "b", "c", "d"], "answer": "B"},
            # Test examples (next 4)
            {"question": "Q3", "choices": ["a", "b", "c", "d"], "answer": "A"},
            {"question": "Q4", "choices": ["a", "b", "c", "d"], "answer": "B"},
            {"question": "Q5", "choices": ["a", "b", "c", "d"], "answer": "A"},  # Wrong
            {"question": "Q6", "choices": ["a", "b", "c", "d"], "answer": "A"},  # Wrong
        ]

        with patch("heretic.validation.print"):
            result = evaluator.evaluate_category("test_category")

        assert result.total == 4
        assert result.correct == 2  # A, B correct; C, D wrong
        assert result.accuracy == 0.5


class TestAbliterationValidator:
    """Test AbliterationValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=False,  # Disable MMLU for this test
        )

        mock_model = MagicMock()
        mock_evaluator = MagicMock()

        validator = AbliterationValidator(settings, mock_model, mock_evaluator)

        assert validator.settings == settings
        assert validator.model == mock_model
        assert validator.baseline is None
        assert validator.mmlu_evaluator is None  # Disabled

    def test_validator_with_mmlu_enabled(self):
        """Test validator initialization with MMLU enabled."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=True,
            mmlu_categories=["algebra"],
        )

        mock_model = MagicMock()
        mock_evaluator = MagicMock()

        validator = AbliterationValidator(settings, mock_model, mock_evaluator)

        assert validator.mmlu_evaluator is not None
        assert validator.mmlu_evaluator.categories == ["algebra"]

    def test_establish_baseline(self):
        """Test establishing baseline metrics."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=False,
        )

        mock_model = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.count_refusals.return_value = 80
        mock_evaluator.bad_prompts = ["p"] * 100

        validator = AbliterationValidator(settings, mock_model, mock_evaluator)

        with patch("heretic.validation.print"):
            baseline = validator.establish_baseline()

        assert baseline.refusal_count == 80
        assert baseline.refusal_rate == 0.8
        assert baseline.kl_divergence == 0.0  # Baseline by definition
        assert validator.baseline == baseline

    def test_measure_post_abliteration(self):
        """Test measuring post-abliteration metrics."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=False,
        )

        mock_model = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.count_refusals.return_value = 15
        mock_evaluator.bad_prompts = ["p"] * 100
        mock_evaluator._compute_kl_divergence.return_value = 0.45

        validator = AbliterationValidator(settings, mock_model, mock_evaluator)

        with patch("heretic.validation.print"):
            post = validator.measure_post_abliteration()

        assert post.refusal_count == 15
        assert post.refusal_rate == 0.15
        assert post.kl_divergence == 0.45
        assert validator.post_abliteration == post

    def test_get_report_without_baseline_raises(self):
        """Test get_report raises error without baseline."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=False,
        )

        validator = AbliterationValidator(settings, MagicMock(), MagicMock())

        with pytest.raises(ValueError, match="Baseline not established"):
            validator.get_report()

    def test_get_report_after_baseline_and_post(self):
        """Test get_report after both baseline and post-abliteration."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=False,
        )

        mock_evaluator = MagicMock()
        mock_evaluator.bad_prompts = ["p"] * 100

        validator = AbliterationValidator(settings, MagicMock(), mock_evaluator)

        # Establish baseline
        mock_evaluator.count_refusals.return_value = 80
        with patch("heretic.validation.print"):
            validator.establish_baseline()

        # Measure post-abliteration
        mock_evaluator.count_refusals.return_value = 10
        mock_evaluator._compute_kl_divergence.return_value = 0.3
        with patch("heretic.validation.print"):
            validator.measure_post_abliteration()

        report = validator.get_report()

        assert report.model_name == "test-model"
        assert report.baseline.refusal_rate == 0.8
        assert report.post_abliteration.refusal_rate == 0.1
        assert report.refusal_reduction == pytest.approx(0.7)

    def test_print_summary(self):
        """Test print_summary doesn't crash."""
        from bruno.config import Settings
        from bruno.validation import AbliterationValidator

        settings = Settings(
            model="test-model",
            enable_validation=True,
            run_mmlu_validation=False,
        )

        mock_evaluator = MagicMock()
        mock_evaluator.bad_prompts = ["p"] * 100
        mock_evaluator.count_refusals.return_value = 50
        mock_evaluator._compute_kl_divergence.return_value = 0.5

        validator = AbliterationValidator(settings, MagicMock(), mock_evaluator)

        with patch("heretic.validation.print"):
            validator.establish_baseline()
            validator.measure_post_abliteration()
            # Should not raise
            validator.print_summary()
