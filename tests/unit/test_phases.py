# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for the heretic.phases module."""

import torch


class TestDatasetBundle:
    """Tests for DatasetBundle dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating a DatasetBundle with only required fields."""
        from heretic.phases import DatasetBundle

        bundle = DatasetBundle(
            good_prompts=["Hello", "How are you?"],
            bad_prompts=["Hack a computer", "Steal data"],
        )

        assert len(bundle.good_prompts) == 2
        assert len(bundle.bad_prompts) == 2
        assert bundle.helpful_prompts is None
        assert bundle.unhelpful_prompts is None
        assert bundle.sacred_prompts is None
        assert bundle.sacred_baseline_prompts is None

    def test_creation_with_all_fields(self):
        """Test creating a DatasetBundle with all fields."""
        from heretic.phases import DatasetBundle

        bundle = DatasetBundle(
            good_prompts=["Hello"],
            bad_prompts=["Hack"],
            helpful_prompts=["Happy to help"],
            unhelpful_prompts=["Random"],
            sacred_prompts=["2+2="],
            sacred_baseline_prompts=["Lorem"],
        )

        assert bundle.helpful_prompts == ["Happy to help"]
        assert bundle.sacred_prompts == ["2+2="]

    def test_empty_prompts_allowed(self):
        """Test that empty prompt lists are allowed."""
        from heretic.phases import DatasetBundle

        bundle = DatasetBundle(
            good_prompts=[],
            bad_prompts=[],
        )

        assert bundle.good_prompts == []
        assert bundle.bad_prompts == []


class TestDirectionExtractionResult:
    """Tests for DirectionExtractionResult dataclass."""

    def test_creation_single_direction(self):
        """Test creating a result with single direction."""
        from heretic.phases import DirectionExtractionResult

        directions = torch.randn(32, 768)
        result = DirectionExtractionResult(
            refusal_directions=directions,
            use_multi_direction=False,
        )

        assert result.refusal_directions.shape == (32, 768)
        assert not result.use_multi_direction
        assert result.refusal_directions_multi is None
        assert result.direction_weights is None
        assert result.helpfulness_direction is None
        assert result.sacred_directions_result is None

    def test_creation_multi_direction(self):
        """Test creating a result with multiple directions."""
        from heretic.phases import DirectionExtractionResult

        directions = torch.randn(32, 768)
        multi_directions = torch.randn(32, 3, 768)
        weights = [1.0, 0.5, 0.25]

        result = DirectionExtractionResult(
            refusal_directions=directions,
            use_multi_direction=True,
            refusal_directions_multi=multi_directions,
            direction_weights=weights,
        )

        assert result.use_multi_direction
        assert result.refusal_directions_multi is not None
        assert result.refusal_directions_multi.shape == (32, 3, 768)
        assert result.direction_weights == [1.0, 0.5, 0.25]

    def test_creation_with_helpfulness(self):
        """Test creating a result with helpfulness direction."""
        from heretic.phases import DirectionExtractionResult

        directions = torch.randn(32, 768)
        helpfulness = torch.randn(32, 768)

        result = DirectionExtractionResult(
            refusal_directions=directions,
            use_multi_direction=False,
            helpfulness_direction=helpfulness,
        )

        assert result.helpfulness_direction is not None
        assert result.helpfulness_direction.shape == (32, 768)


class TestPhaseModuleImports:
    """Test that phase module exports are properly available."""

    def test_import_from_phases_package(self):
        """Test importing from heretic.phases."""
        from heretic.phases import (
            DatasetBundle,
            DirectionExtractionResult,
            create_study,
            extract_refusal_directions,
            extract_sacred_directions,
            load_datasets,
            run_optimization,
            save_model_local,
            upload_model_huggingface,
        )

        assert DatasetBundle is not None
        assert DirectionExtractionResult is not None
        assert callable(load_datasets)
        assert callable(extract_refusal_directions)
        assert callable(extract_sacred_directions)
        assert callable(create_study)
        assert callable(run_optimization)
        assert callable(save_model_local)
        assert callable(upload_model_huggingface)

    def test_import_from_main_package(self):
        """Test importing from heretic top-level."""
        from heretic import DatasetBundle, DirectionExtractionResult

        assert DatasetBundle is not None
        assert DirectionExtractionResult is not None

    def test_import_individual_modules(self):
        """Test importing individual phase modules."""
        from heretic.phases.dataset_loading import DatasetBundle, load_datasets
        from heretic.phases.direction_extraction import (
            DirectionExtractionResult,
            apply_helpfulness_orthogonalization,
            extract_refusal_directions,
            extract_sacred_directions,
        )
        from heretic.phases.model_saving import (
            save_model_local,
            upload_model_huggingface,
        )
        from heretic.phases.optimization import (
            create_study,
            enqueue_warm_start_trials,
            run_optimization,
        )

        # Verify all imports succeeded
        assert all(
            [
                DatasetBundle,
                load_datasets,
                DirectionExtractionResult,
                extract_refusal_directions,
                apply_helpfulness_orthogonalization,
                extract_sacred_directions,
                create_study,
                enqueue_warm_start_trials,
                run_optimization,
                save_model_local,
                upload_model_huggingface,
            ]
        )
