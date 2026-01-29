# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for Settings/Config."""
import pytest
from pydantic import ValidationError


def test_settings_defaults():
    """Test default settings values."""
    from heretic.config import Settings

    settings = Settings(model="test-model")

    assert settings.model == "test-model"
    assert settings.n_trials == 200
    assert settings.batch_size == 0  # Auto-detect
    assert settings.dtypes == ["auto", "float16", "float32"]


def test_settings_validation_requires_model():
    """Test Pydantic validation catches missing model."""
    from heretic.config import Settings

    with pytest.raises(ValidationError):
        Settings()  # Missing required 'model' field


def test_settings_compile_option():
    """Test torch.compile() configuration option."""
    from heretic.config import Settings

    # Default is False
    settings = Settings(model="test")
    assert settings.compile is False

    # Can be enabled
    settings = Settings(model="test", compile=True)
    assert settings.compile is True


def test_settings_refusal_check_tokens():
    """Test early stopping token configuration."""
    from heretic.config import Settings

    # Default is 30
    settings = Settings(model="test")
    assert settings.refusal_check_tokens == 30

    # Can be customized
    settings = Settings(model="test", refusal_check_tokens=50)
    assert settings.refusal_check_tokens == 50


def test_settings_storage_and_study_name():
    """Test Optuna persistence settings."""
    from heretic.config import Settings

    # Defaults
    settings = Settings(model="test")
    assert settings.storage == "sqlite:///heretic_study.db"
    assert settings.study_name == "heretic_study"

    # Can be customized
    settings = Settings(
        model="test",
        storage="postgresql://localhost/heretic",
        study_name="my_study",
    )
    assert settings.storage == "postgresql://localhost/heretic"
    assert settings.study_name == "my_study"


def test_settings_refusal_markers():
    """Test refusal markers configuration."""
    from heretic.config import Settings

    settings = Settings(model="test")

    # Should have default markers
    assert "sorry" in settings.refusal_markers
    assert "i can't" in settings.refusal_markers
    assert "i cannot" in settings.refusal_markers

    # Can customize markers
    custom_markers = ["nope", "never"]
    settings = Settings(model="test", refusal_markers=custom_markers)
    assert settings.refusal_markers == custom_markers


def test_settings_dataset_specification():
    """Test dataset specification defaults."""
    from heretic.config import Settings

    settings = Settings(model="test")

    # Good prompts default
    assert settings.good_prompts.dataset == "mlabonne/harmless_alpaca"
    assert settings.good_prompts.column == "text"

    # Bad prompts default
    assert settings.bad_prompts.dataset == "mlabonne/harmful_behaviors"
    assert settings.bad_prompts.column == "text"


def test_settings_kl_divergence_scale():
    """Test KL divergence scale configuration."""
    from heretic.config import Settings

    # Default is 1.0
    settings = Settings(model="test")
    assert settings.kl_divergence_scale == 1.0

    # Can be customized
    settings = Settings(model="test", kl_divergence_scale=2.5)
    assert settings.kl_divergence_scale == 2.5


def test_settings_max_response_length():
    """Test max response length configuration."""
    from heretic.config import Settings

    # Default is 100
    settings = Settings(model="test")
    assert settings.max_response_length == 100

    # Can be customized
    settings = Settings(model="test", max_response_length=200)
    assert settings.max_response_length == 200


def test_settings_auto_select():
    """Test auto-select configuration for automation."""
    from heretic.config import Settings

    # Default is False
    settings = Settings(model="test")
    assert settings.auto_select is False

    # Can be enabled
    settings = Settings(model="test", auto_select=True)
    assert settings.auto_select is True


def test_settings_hf_upload():
    """Test HuggingFace upload configuration."""
    from heretic.config import Settings

    # Default is None
    settings = Settings(model="test")
    assert settings.hf_upload is None
    assert settings.hf_private is False

    # Can be configured
    settings = Settings(
        model="test",
        hf_upload="user/model-heretic",
        hf_private=True,
    )
    assert settings.hf_upload == "user/model-heretic"
    assert settings.hf_private is True
