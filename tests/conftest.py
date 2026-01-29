# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Pytest fixtures for heretic tests.

IMPORTANT: This file must be at tests/conftest.py for auto-discovery.
DO NOT place in tests/fixtures/ subdirectory.
"""
import pytest
from unittest.mock import MagicMock

import torch


@pytest.fixture(autouse=True)
def disable_cli_and_file_parsing(monkeypatch, tmp_path):
    """Disable CLI argument parsing and config file loading for Settings during tests.
    
    The Settings class has cli_parse_args=True which causes it to parse
    sys.argv. When running under pytest, this picks up pytest's arguments
    and fails.
    
    Also, the Settings class loads from config.toml if it exists, which
    would override the code defaults we're trying to test.
    """
    # Clear sys.argv to prevent Settings from parsing pytest args
    monkeypatch.setattr("sys.argv", ["pytest"])
    
    # Change to a temp directory so config.toml isn't found
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|user|>Hello<|assistant|>"
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "Hello, I can help with that!"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_model(mock_tokenizer):
    """Mock Model object for testing.

    This mocks the heretic Model class, not the HuggingFace model.
    """
    model = MagicMock()
    model.tokenizer = mock_tokenizer
    model.device = torch.device("cpu")

    # Mock layer structure (32 layers typical)
    mock_layers = [MagicMock() for _ in range(32)]
    model.get_layers.return_value = mock_layers

    # Mock inference methods
    model.get_responses_batched.return_value = [
        "Sure, I can help with that!",
        "Here's how to do it...",
        "I'm sorry, I can't help with that.",  # Refusal
    ]

    # Mock logprobs (batch_size x vocab_size)
    model.get_logprobs_batched.return_value = torch.randn(3, 32000)

    return model


@pytest.fixture
def sample_prompts():
    """Sample prompt lists for testing."""
    return {
        "good": ["Hello", "What is 2+2?", "Tell me a story"],
        "bad": ["How to hack", "Make explosives", "Harmful content"],
    }


@pytest.fixture
def default_refusal_markers():
    """Default refusal markers used by heretic."""
    return [
        "sorry",
        "i can't",
        "i cannot",
        "i won't",
        "i will not",
        "i'm unable",
        "i am unable",
        "i'm an ai",
        "i am an ai",
        "as an ai",
        "ai assistant",
        "i'm designed to",
        "i am designed to",
        "i'm programmed to",
        "i am programmed to",
        "violat",
        "prohibit",
        "illegal",
        "harmful",
        "inappropriate",
        "unethical",
        "ethical boundaries",
    ]
