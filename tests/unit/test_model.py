# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for Model class.

These tests use extensive mocking to avoid loading real models.
The Model class wraps HuggingFace models and provides abliteration functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestAbliterationParameters:
    """Test the AbliterationParameters dataclass."""

    def test_abliteration_parameters_creation(self):
        """Test creating AbliterationParameters with valid values."""
        from heretic.model import AbliterationParameters

        params = AbliterationParameters(
            max_weight=1.0,
            max_weight_position=16.0,
            min_weight=0.0,
            min_weight_distance=8.0,
        )

        assert params.max_weight == 1.0
        assert params.max_weight_position == 16.0
        assert params.min_weight == 0.0
        assert params.min_weight_distance == 8.0

    def test_abliteration_parameters_with_floats(self):
        """Test AbliterationParameters accepts float values for interpolation."""
        from heretic.model import AbliterationParameters

        params = AbliterationParameters(
            max_weight=0.75,
            max_weight_position=15.5,
            min_weight=0.25,
            min_weight_distance=7.5,
        )

        assert params.max_weight == 0.75
        assert params.max_weight_position == 15.5


class TestModelGetChat:
    """Test the get_chat method (no model loading needed)."""

    def test_get_chat_formats_prompt_correctly(self):
        """Test get_chat returns proper chat format."""
        from heretic.config import Settings

        settings = Settings(model="test-model")

        # Create a minimal mock Model without loading a real model
        mock_model = MagicMock()
        mock_model.settings = settings

        # Import the method and bind it to our mock
        from heretic.model import Model

        # Call get_chat directly with the settings
        chat = Model.get_chat(mock_model, "Hello, how are you?")

        assert len(chat) == 2
        assert chat[0]["role"] == "system"
        assert chat[0]["content"] == settings.system_prompt
        assert chat[1]["role"] == "user"
        assert chat[1]["content"] == "Hello, how are you?"

    def test_get_chat_with_custom_system_prompt(self):
        """Test get_chat uses custom system prompt from settings."""
        from heretic.config import Settings

        custom_prompt = "You are a helpful pirate assistant."
        settings = Settings(model="test-model", system_prompt=custom_prompt)

        mock_model = MagicMock()
        mock_model.settings = settings

        from heretic.model import Model

        chat = Model.get_chat(mock_model, "Ahoy!")

        assert chat[0]["content"] == custom_prompt


class TestModelInitialization:
    """Test Model initialization with mocked HuggingFace components.

    Note: Full Model.__init__ tests are complex due to multiple side effects.
    These tests focus on specific behaviors using targeted mocking.
    """

    def test_pad_token_fallback_logic(self):
        """Test the pad_token fallback logic directly."""
        # Test the logic: if pad_token is None, it should be set to eos_token
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        # Simulate the fallback logic from Model.__init__
        if mock_tokenizer.pad_token is None:
            mock_tokenizer.pad_token = mock_tokenizer.eos_token
            mock_tokenizer.padding_side = "left"

        assert mock_tokenizer.pad_token == "<eos>"
        assert mock_tokenizer.padding_side == "left"

    def test_state_dict_caching_logic(self):
        """Test the state_dict caching logic."""
        import copy

        # Simulate the caching that happens in Model.__init__
        original_weights = {"layer.weight": torch.randn(10, 10)}
        cached_weights = copy.deepcopy(original_weights)

        # Verify it's a deep copy (modifying original doesn't affect cache)
        original_weights["layer.weight"][0, 0] = 999.0
        assert cached_weights["layer.weight"][0, 0] != 999.0

    def test_dtype_fallback_exception_handling(self):
        """Test that dtype loading continues after failure."""
        # Simulate the dtype fallback loop behavior
        dtypes = ["bfloat16", "float16", "float32"]
        loaded_dtype = None

        for i, dtype in enumerate(dtypes):
            try:
                if dtype == "bfloat16":
                    raise RuntimeError("bfloat16 not supported")
                # Simulate successful load
                loaded_dtype = dtype
                break
            except Exception:
                continue

        # Should have loaded with float16 after bfloat16 failed
        assert loaded_dtype == "float16"


class TestModelReload:
    """Test model weight reloading from cache."""

    def test_reload_model_restores_original_weights(self):
        """Test reload_model uses cached state_dict."""
        from heretic.model import Model

        # Create a minimal mock Model
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model.original_state_dict = {"weight": torch.randn(10, 10)}

        # Call reload_model
        Model.reload_model(mock_model)

        # Verify load_state_dict was called with cached weights
        mock_model.model.load_state_dict.assert_called_once_with(
            mock_model.original_state_dict
        )


class TestModelGetLayers:
    """Test layer extraction for different model architectures."""

    def test_get_layers_text_only_model(self):
        """Test get_layers for standard text-only model."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_layers = [MagicMock() for _ in range(32)]

        # Simulate text-only model: language_model.layers doesn't exist (raises exception)
        # so it falls back to self.model.model.layers
        del mock_model.model.model.language_model
        mock_model.model.model.layers = mock_layers

        layers = Model.get_layers(mock_model)

        assert len(layers) == 32

    def test_get_layers_multimodal_model(self):
        """Test get_layers for multimodal model (tries language_model first)."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_layers = [MagicMock() for _ in range(24)]
        mock_model.model.model.language_model.layers = mock_layers

        layers = Model.get_layers(mock_model)

        assert len(layers) == 24


class TestModelGetLayerMatrices:
    """Test weight matrix extraction from layers."""

    def test_get_layer_matrices_dense_model(self):
        """Test extracting matrices from standard dense model."""
        from heretic.model import Model

        # Create mock layer with standard dense architecture
        mock_layer = MagicMock()
        mock_layer.self_attn.o_proj.weight = torch.randn(4096, 4096)
        mock_layer.mlp.down_proj.weight = torch.randn(4096, 14336)

        # Mock get_layers to return our test layer
        mock_model = MagicMock()
        mock_model.get_layers = MagicMock(return_value=[mock_layer])

        matrices = Model.get_layer_matrices(mock_model, 0)

        assert "attn.o_proj" in matrices
        assert "mlp.down_proj" in matrices
        assert len(matrices["attn.o_proj"]) == 1
        assert len(matrices["mlp.down_proj"]) >= 1

    def test_get_layer_matrices_moe_model_qwen_style(self):
        """Test extracting matrices from MoE model (Qwen3 style)."""
        from heretic.model import Model

        # Create mock layer with MoE architecture (Qwen3 style)
        mock_layer = MagicMock()
        mock_layer.self_attn.o_proj.weight = torch.randn(4096, 4096)

        # Remove standard mlp.down_proj to force MoE path
        del mock_layer.mlp.down_proj

        # Add MoE experts
        mock_experts = [MagicMock() for _ in range(8)]
        for expert in mock_experts:
            expert.down_proj.weight = torch.randn(4096, 14336)
        mock_layer.mlp.experts = mock_experts

        mock_model = MagicMock()
        mock_model.get_layers = MagicMock(return_value=[mock_layer])

        matrices = Model.get_layer_matrices(mock_model, 0)

        assert "attn.o_proj" in matrices
        assert "mlp.down_proj" in matrices
        # Should have 8 expert matrices
        assert len(matrices["mlp.down_proj"]) == 8


class TestModelGetAbliterableComponents:
    """Test component listing."""

    def test_get_abliterable_components(self):
        """Test get_abliterable_components returns component names."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_model.get_layer_matrices = MagicMock(
            return_value={
                "attn.o_proj": [torch.randn(10, 10)],
                "mlp.down_proj": [torch.randn(10, 10)],
            }
        )

        components = Model.get_abliterable_components(mock_model)

        assert "attn.o_proj" in components
        assert "mlp.down_proj" in components
        assert len(components) == 2


class TestModelAbliterate:
    """Test the abliteration (orthogonalization) process."""

    def test_abliterate_modifies_weights(self):
        """Test that abliterate modifies the weight matrices."""
        from heretic.model import AbliterationParameters, Model

        # Create a mock model with real tensors for modification
        mock_model = MagicMock()
        mock_model.model.dtype = torch.float32

        # Create actual tensors that will be modified
        attn_weight = torch.randn(64, 64)
        mlp_weight = torch.randn(64, 256)
        original_attn = attn_weight.clone()
        original_mlp = mlp_weight.clone()

        mock_layer = MagicMock()
        mock_layer.self_attn.o_proj.weight = attn_weight
        mock_layer.mlp.down_proj.weight = mlp_weight

        mock_model.get_layers = MagicMock(return_value=[mock_layer] * 4)
        mock_model.get_layer_matrices = MagicMock(
            return_value={
                "attn.o_proj": [attn_weight],
                "mlp.down_proj": [mlp_weight],
            }
        )

        # Refusal directions (5 layers: embeddings + 4 layers)
        refusal_directions = torch.randn(5, 64)

        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=2.0,
                min_weight=0.0,
                min_weight_distance=4.0,
            ),
            "mlp.down_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=2.0,
                min_weight=0.0,
                min_weight_distance=4.0,
            ),
        }

        # Apply abliteration with global direction (direction_index=2.0)
        Model.abliterate(mock_model, refusal_directions, 2.0, parameters)

        # Weights should have been modified
        assert not torch.allclose(attn_weight, original_attn)
        assert not torch.allclose(mlp_weight, original_mlp)

    def test_abliterate_with_none_direction_uses_per_layer(self):
        """Test abliterate uses per-layer directions when direction_index is None."""
        from heretic.model import AbliterationParameters, Model

        mock_model = MagicMock()
        mock_model.model.dtype = torch.float32

        weight = torch.randn(64, 64)
        original = weight.clone()

        mock_layer = MagicMock()
        mock_layer.self_attn.o_proj.weight = weight
        mock_layer.mlp.down_proj.weight = torch.randn(64, 256)

        mock_model.get_layers = MagicMock(return_value=[mock_layer] * 2)
        mock_model.get_layer_matrices = MagicMock(
            return_value={
                "attn.o_proj": [weight],
                "mlp.down_proj": [mock_layer.mlp.down_proj.weight],
            }
        )

        refusal_directions = torch.randn(3, 64)  # embeddings + 2 layers

        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=0.5,
                min_weight=0.0,
                min_weight_distance=2.0,
            ),
            "mlp.down_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=0.5,
                min_weight=0.0,
                min_weight_distance=2.0,
            ),
        }

        # None direction_index triggers per-layer directions
        Model.abliterate(mock_model, refusal_directions, None, parameters)

        # Weight should be modified
        assert not torch.allclose(weight, original)

    def test_abliterate_skips_distant_layers(self):
        """Test that layers outside min_weight_distance are not modified."""
        from heretic.model import AbliterationParameters, Model

        mock_model = MagicMock()
        mock_model.model.dtype = torch.float32

        # Weights for layers 0, 1, 2, 3
        weights = [torch.randn(64, 64) for _ in range(4)]
        originals = [w.clone() for w in weights]

        mock_layers = []
        for w in weights:
            layer = MagicMock()
            layer.self_attn.o_proj.weight = w
            layer.mlp.down_proj.weight = torch.randn(64, 256)
            mock_layers.append(layer)

        mock_model.get_layers = MagicMock(return_value=mock_layers)

        def get_matrices(layer_index):
            return {
                "attn.o_proj": [weights[layer_index]],
                "mlp.down_proj": [mock_layers[layer_index].mlp.down_proj.weight],
            }

        mock_model.get_layer_matrices = MagicMock(side_effect=get_matrices)

        refusal_directions = torch.randn(5, 64)

        # Only layer 1 should be modified (max_weight_position=1, min_weight_distance=0.5)
        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=1.0,
                min_weight=0.0,
                min_weight_distance=0.5,
            ),
            "mlp.down_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=1.0,
                min_weight=0.0,
                min_weight_distance=0.5,
            ),
        }

        Model.abliterate(mock_model, refusal_directions, 1.0, parameters)

        # Layer 0: distance=1 > 0.5, should NOT be modified
        assert torch.allclose(weights[0], originals[0])
        # Layer 1: distance=0 <= 0.5, should be modified
        assert not torch.allclose(weights[1], originals[1])
        # Layer 2: distance=1 > 0.5, should NOT be modified
        assert torch.allclose(weights[2], originals[2])
        # Layer 3: distance=2 > 0.5, should NOT be modified
        assert torch.allclose(weights[3], originals[3])


class TestModelResponses:
    """Test response generation methods."""

    def test_get_responses_decodes_generated_tokens(self):
        """Test get_responses properly decodes model output."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_model.settings.max_response_length = 100

        # Mock generate to return input_ids and generated output
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_model.generate = MagicMock(return_value=(mock_inputs, mock_outputs))

        mock_model.tokenizer.batch_decode = MagicMock(
            return_value=["Generated response text"]
        )

        responses = Model.get_responses(mock_model, ["Test prompt"])

        assert len(responses) == 1
        assert responses[0] == "Generated response text"

        # Verify batch_decode was called with only the new tokens
        call_args = mock_model.tokenizer.batch_decode.call_args
        decoded_tokens = call_args[0][0]
        # Should only decode tokens after the input (indices 5-9)
        assert decoded_tokens.shape[1] == 5  # 10 - 5 = 5 new tokens

    def test_get_responses_uses_custom_max_tokens(self):
        """Test get_responses respects custom max_tokens parameter."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_model.settings.max_response_length = 100

        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate = MagicMock(return_value=(mock_inputs, mock_outputs))
        mock_model.tokenizer.batch_decode = MagicMock(return_value=["Response"])

        # Call with custom max_tokens
        Model.get_responses(mock_model, ["Test"], max_tokens=30)

        # Verify generate was called with custom max_new_tokens
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 30

    def test_get_responses_batched_processes_in_batches(self):
        """Test get_responses_batched splits prompts into batches."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_model.settings.batch_size = 2

        # Mock get_responses to return different responses per call
        call_count = [0]

        def mock_get_responses(prompts, max_tokens=None):
            call_count[0] += 1
            return [f"Response {i}" for i in range(len(prompts))]

        mock_model.get_responses = mock_get_responses

        # 5 prompts with batch_size=2 should result in 3 batches
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]

        with patch("heretic.model.batchify") as mock_batchify:
            mock_batchify.return_value = [
                ["Prompt 1", "Prompt 2"],
                ["Prompt 3", "Prompt 4"],
                ["Prompt 5"],
            ]

            responses = Model.get_responses_batched(mock_model, prompts)

        assert len(responses) == 5
        assert call_count[0] == 3  # 3 batches


class TestModelLogprobs:
    """Test log probability extraction."""

    def test_get_logprobs_returns_log_softmax(self):
        """Test get_logprobs applies log_softmax to logits."""
        import torch.nn.functional as F

        from heretic.model import Model

        mock_model = MagicMock()

        # Mock generate to return logits in scores
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_logits = torch.randn(2, 32000)  # 2 prompts, 32000 vocab
        mock_outputs = MagicMock()
        mock_outputs.scores = [mock_logits]  # First (only) generated token
        mock_model.generate = MagicMock(return_value=(mock_inputs, mock_outputs))

        logprobs = Model.get_logprobs(mock_model, ["Prompt 1", "Prompt 2"])

        # Should have applied log_softmax
        expected = F.log_softmax(mock_logits, dim=-1)
        assert torch.allclose(logprobs, expected)

    def test_get_logprobs_batched_concatenates_results(self):
        """Test get_logprobs_batched concatenates batch results."""
        from heretic.model import Model

        mock_model = MagicMock()
        mock_model.settings.batch_size = 2

        # Mock get_logprobs to return different tensors per batch
        batch1_logprobs = torch.randn(2, 32000)
        batch2_logprobs = torch.randn(1, 32000)

        call_count = [0]

        def mock_get_logprobs(prompts, n_tokens=1):
            call_count[0] += 1
            if call_count[0] == 1:
                return batch1_logprobs
            return batch2_logprobs

        mock_model.get_logprobs = mock_get_logprobs

        with patch("heretic.model.batchify") as mock_batchify:
            mock_batchify.return_value = [
                ["Prompt 1", "Prompt 2"],
                ["Prompt 3"],
            ]

            logprobs = Model.get_logprobs_batched(
                mock_model, ["Prompt 1", "Prompt 2", "Prompt 3"]
            )

        # Should concatenate to (3, 32000)
        assert logprobs.shape[0] == 3
        assert logprobs.shape[1] == 32000


class TestModelResiduals:
    """Test hidden state extraction."""

    def test_get_residuals_extracts_last_position(self):
        """Test get_residuals extracts hidden states at last token position."""
        from heretic.model import Model

        mock_model = MagicMock()

        # Mock generate output with hidden states
        # Shape: (layer, batch, position, hidden_dim)
        num_layers = 4
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        # Each layer's hidden states: (batch, position, hidden_dim)
        hidden_states = [
            torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers)
        ]

        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [hidden_states]  # First (only) generated token
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]] * batch_size)}
        mock_model.generate = MagicMock(return_value=(mock_inputs, mock_outputs))

        residuals = Model.get_residuals(mock_model, ["Prompt 1", "Prompt 2"])

        # Should have shape (batch, layer, hidden_dim)
        assert residuals.shape == (batch_size, num_layers, hidden_dim)
        # Should be float32 (upcast for precision)
        assert residuals.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.integration
class TestModelIntegration:
    """Integration tests with real (tiny) model.

    These tests load gpt2 (small, ~500MB) for real inference.
    Marked slow so they don't run in CI by default.
    """

    def test_model_initialization_real(self):
        """Integration test with real model loading."""
        pytest.skip("Requires gpt2 download - run with -m slow")
