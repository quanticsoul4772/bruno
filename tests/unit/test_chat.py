"""Tests for bruno.chat module."""

import os
from unittest.mock import MagicMock, patch


class TestChatEnvironmentSetup:
    """Test that environment variables are set correctly."""

    def test_symlinks_disabled_on_import(self):
        """Test HF_HUB_DISABLE_SYMLINKS is set before imports."""
        # Import cli module (which sets env vars)
        from bruno import cli  # noqa: F401

        assert os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1"
        assert os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING") == "1"


class TestModelSelector:
    """Test interactive model selector."""

    def test_select_model_returns_default_on_no_models(self):
        """Test selector returns default when no models found."""
        from bruno.chat import select_model_interactive

        with patch("bruno.chat.list_models", return_value=[]):
            result = select_model_interactive(default="test-model")
            assert result == "test-model"

    def test_select_model_handles_api_error(self):
        """Test selector handles API errors gracefully."""
        from bruno.chat import select_model_interactive

        with patch("bruno.chat.list_models", side_effect=Exception("API error")):
            result = select_model_interactive(default="fallback-model")
            assert result == "fallback-model"


class TestRunChat:
    """Test run_chat function."""

    @patch("bruno.chat.BrunoChat")
    @patch("bruno.chat.select_model_interactive")
    def test_run_chat_uses_selector_when_no_model(self, mock_selector, mock_bruno_chat):
        """Test run_chat calls selector when model is None."""
        from bruno.chat import run_chat

        mock_selector.return_value = "selected-model"
        mock_chat_instance = MagicMock()
        mock_bruno_chat.return_value = mock_chat_instance

        run_chat(model_name=None)

        # Verify selector was called
        mock_selector.assert_called_once()

        # Verify BrunoChat was created with selected model
        mock_bruno_chat.assert_called_once()
        assert mock_bruno_chat.call_args[1]["model_name"] == "selected-model"

    @patch("bruno.chat.BrunoChat")
    def test_run_chat_skips_selector_when_model_provided(self, mock_bruno_chat):
        """Test run_chat skips selector when model is specified."""
        from bruno.chat import run_chat

        mock_chat_instance = MagicMock()
        mock_bruno_chat.return_value = mock_chat_instance

        run_chat(model_name="direct-model")

        # Verify BrunoChat was created with provided model
        mock_bruno_chat.assert_called_once()
        assert mock_bruno_chat.call_args[1]["model_name"] == "direct-model"


class TestBrunoChatClass:
    """Test BrunoChat class."""

    def test_bruno_chat_init_sets_history_with_system_prompt(self):
        """Test system prompt is added to history."""
        from bruno.chat import BrunoChat

        with (
            patch("bruno.chat.AutoModelForCausalLM"),
            patch("bruno.chat.AutoTokenizer"),
        ):
            chat = BrunoChat(
                model_name="test-model", system_prompt="You are a coding assistant"
            )

            assert len(chat.history) == 1
            assert chat.history[0]["role"] == "system"
            assert chat.history[0]["content"] == "You are a coding assistant"

    def test_bruno_chat_init_empty_history_without_system_prompt(self):
        """Test history is empty without system prompt."""
        from bruno.chat import BrunoChat

        with (
            patch("bruno.chat.AutoModelForCausalLM"),
            patch("bruno.chat.AutoTokenizer"),
        ):
            chat = BrunoChat(model_name="test-model")

            assert len(chat.history) == 0
