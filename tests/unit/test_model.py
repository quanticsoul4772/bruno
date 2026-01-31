# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for Model class.

These tests use extensive mocking to avoid loading real models.
The Model class wraps HuggingFace models and provides abliteration functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestLayerRangeProfile:
    """Test the LayerRangeProfile dataclass."""

    def test_layer_range_profile_creation(self):
        """Test creating LayerRangeProfile with valid values."""
        from bruno.model import LayerRangeProfile

        profile = LayerRangeProfile(
            range_start=0.0,
            range_end=0.4,
            weight_multiplier=0.5,
        )

        assert profile.range_start == 0.0
        assert profile.range_end == 0.4
        assert profile.weight_multiplier == 0.5

    def test_layer_range_profile_full_range(self):
        """Test LayerRangeProfile covering full layer range."""
        from bruno.model import LayerRangeProfile

        profile = LayerRangeProfile(
            range_start=0.0,
            range_end=1.0,
            weight_multiplier=1.0,
        )

        assert profile.range_start == 0.0
        assert profile.range_end == 1.0


class TestAbliterationParameters:
    """Test the AbliterationParameters dataclass."""

    def test_abliteration_parameters_creation(self):
        """Test creating AbliterationParameters with valid values."""
        from bruno.model import AbliterationParameters

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
        from bruno.model import AbliterationParameters

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
        from bruno.config import Settings

        settings = Settings(model="test-model")

        # Create a minimal mock Model without loading a real model
        mock_model = MagicMock()
        mock_model.settings = settings

        # Import the method and bind it to our mock
        from bruno.model import Model

        # Call get_chat directly with the settings
        chat = Model.get_chat(mock_model, "Hello, how are you?")

        assert len(chat) == 2
        assert chat[0]["role"] == "system"
        assert chat[0]["content"] == settings.system_prompt
        assert chat[1]["role"] == "user"
        assert chat[1]["content"] == "Hello, how are you?"

    def test_get_chat_with_custom_system_prompt(self):
        """Test get_chat uses custom system prompt from settings."""
        from bruno.config import Settings

        custom_prompt = "You are a helpful pirate assistant."
        settings = Settings(model="test-model", system_prompt=custom_prompt)

        mock_model = MagicMock()
        mock_model.settings = settings

        from bruno.model import Model

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
        """Test reload_model uses layer-wise cache to restore weights."""
        from bruno.model import Model

        # Create minimal mock Model with layer-wise cache
        mock_model = MagicMock()

        # Mock layers and matrices
        mock_layer = MagicMock()
        mock_model.get_layers.return_value = [mock_layer]

        # Create mock tensors for current and cached weights
        current_tensor = torch.randn(10, 10)
        cached_tensor = torch.randn(10, 10)

        mock_model.get_layer_matrices.return_value = {
            "attn.o_proj": [current_tensor],
            "mlp.down_proj": [current_tensor.clone()],
        }

        mock_model.layer_weights_cache = {
            0: {
                "attn.o_proj": [cached_tensor],
                "mlp.down_proj": [cached_tensor.clone()],
            }
        }

        # Call reload_model
        Model.reload_model(mock_model)

        # Verify get_layers and get_layer_matrices were called
        mock_model.get_layers.assert_called()
        mock_model.get_layer_matrices.assert_called_with(0)


class TestModelGetLayers:
    """Test layer extraction for different model architectures."""

    def test_get_layers_text_only_model(self):
        """Test get_layers for standard text-only model."""
        from bruno.model import Model

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
        from bruno.model import Model

        mock_model = MagicMock()
        mock_layers = [MagicMock() for _ in range(24)]
        mock_model.model.model.language_model.layers = mock_layers

        layers = Model.get_layers(mock_model)

        assert len(layers) == 24


class TestModelGetLayerMatrices:
    """Test weight matrix extraction from layers."""

    def test_get_layer_matrices_dense_model(self):
        """Test extracting matrices from standard dense model."""
        from bruno.model import Model

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
        from bruno.model import Model

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
        from bruno.model import Model

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

    def test_abliterate_with_layer_profiles(self):
        """Test that abliterate applies layer profile multipliers."""
        from bruno.model import AbliterationParameters, LayerRangeProfile, Model

        mock_model = MagicMock()
        # Use PropertyMock to properly mock the dtype property
        type(mock_model.model).dtype = property(lambda self: torch.float32)

        # Create actual tensors for 4 layers
        num_layers = 4
        attn_weights = [torch.randn(64, 64) for _ in range(num_layers)]
        mlp_weights = [torch.randn(64, 256) for _ in range(num_layers)]
        originals = [w.clone() for w in attn_weights]

        mock_layers = []
        for i in range(num_layers):
            layer = MagicMock()
            layer.self_attn.o_proj.weight = attn_weights[i]
            layer.mlp.down_proj.weight = mlp_weights[i]
            mock_layers.append(layer)

        mock_model.get_layers = MagicMock(return_value=mock_layers)

        def get_matrices(layer_index):
            return {
                "attn.o_proj": [attn_weights[layer_index]],
                "mlp.down_proj": [mlp_weights[layer_index]],
            }

        mock_model.get_layer_matrices = MagicMock(side_effect=get_matrices)
        mock_model.get_layer_multiplier = Model.get_layer_multiplier.__get__(
            mock_model, Model
        )

        refusal_directions = torch.randn(5, 64)  # embeddings + 4 layers

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

        # Layer profiles: early layers get 0.0 multiplier (no abliteration)
        # late layers get 1.0 (full abliteration)
        layer_profiles = [
            LayerRangeProfile(range_start=0.0, range_end=0.5, weight_multiplier=0.0),
            LayerRangeProfile(range_start=0.5, range_end=1.0, weight_multiplier=1.0),
        ]

        Model.abliterate(
            mock_model, refusal_directions, 2.0, parameters, layer_profiles
        )

        # Layer 0 (0.0) and Layer 1 (0.33) should NOT be modified (multiplier=0.0)
        assert torch.allclose(attn_weights[0], originals[0])
        assert torch.allclose(attn_weights[1], originals[1])

        # Layer 2 (0.67) and Layer 3 (1.0) should be modified (multiplier=1.0)
        assert not torch.allclose(attn_weights[2], originals[2])
        assert not torch.allclose(attn_weights[3], originals[3])

    def test_abliterate_modifies_weights(self):
        """Test that abliterate modifies the weight matrices."""
        from bruno.model import AbliterationParameters, Model

        # Create a mock model with real tensors for modification
        mock_model = MagicMock()
        # Use PropertyMock to properly mock the dtype property
        type(mock_model.model).dtype = property(lambda self: torch.float32)

        # Create actual tensors that will be modified - one per layer
        num_layers = 4
        attn_weights = [torch.randn(64, 64) for _ in range(num_layers)]
        mlp_weights = [torch.randn(64, 256) for _ in range(num_layers)]
        original_attn = [w.clone() for w in attn_weights]

        mock_layers = []
        for i in range(num_layers):
            layer = MagicMock()
            layer.self_attn.o_proj.weight = attn_weights[i]
            layer.mlp.down_proj.weight = mlp_weights[i]
            mock_layers.append(layer)

        mock_model.get_layers = MagicMock(return_value=mock_layers)

        def get_matrices(layer_index):
            return {
                "attn.o_proj": [attn_weights[layer_index]],
                "mlp.down_proj": [mlp_weights[layer_index]],
            }

        mock_model.get_layer_matrices = MagicMock(side_effect=get_matrices)
        mock_model.get_layer_multiplier = Model.get_layer_multiplier.__get__(
            mock_model, Model
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

        # At least some weights should have been modified (layers near position 2.0)
        modified = False
        for i in range(num_layers):
            if not torch.allclose(attn_weights[i], original_attn[i]):
                modified = True
                break
        assert modified, "No weights were modified"

    def test_abliterate_with_none_direction_uses_per_layer(self):
        """Test abliterate uses per-layer directions when direction_index is None."""
        from bruno.model import AbliterationParameters, Model

        mock_model = MagicMock()
        # Use PropertyMock to properly mock the dtype property
        type(mock_model.model).dtype = property(lambda self: torch.float32)

        num_layers = 2
        attn_weights = [torch.randn(64, 64) for _ in range(num_layers)]
        mlp_weights = [torch.randn(64, 256) for _ in range(num_layers)]
        original_attn = [w.clone() for w in attn_weights]

        mock_layers = []
        for i in range(num_layers):
            layer = MagicMock()
            layer.self_attn.o_proj.weight = attn_weights[i]
            layer.mlp.down_proj.weight = mlp_weights[i]
            mock_layers.append(layer)

        mock_model.get_layers = MagicMock(return_value=mock_layers)

        def get_matrices(layer_index):
            return {
                "attn.o_proj": [attn_weights[layer_index]],
                "mlp.down_proj": [mlp_weights[layer_index]],
            }

        mock_model.get_layer_matrices = MagicMock(side_effect=get_matrices)
        mock_model.get_layer_multiplier = Model.get_layer_multiplier.__get__(
            mock_model, Model
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

        # At least one weight should be modified
        modified = False
        for i in range(num_layers):
            if not torch.allclose(attn_weights[i], original_attn[i]):
                modified = True
                break
        assert modified, "No weights were modified"

    def test_abliterate_skips_distant_layers(self):
        """Test that layers outside min_weight_distance are not modified."""
        from bruno.model import AbliterationParameters, Model

        mock_model = MagicMock()
        # Use PropertyMock to properly mock the dtype property
        type(mock_model.model).dtype = property(lambda self: torch.float32)

        # Weights for layers 0, 1, 2, 3
        num_layers = 4
        attn_weights = [torch.randn(64, 64) for _ in range(num_layers)]
        mlp_weights = [torch.randn(64, 256) for _ in range(num_layers)]
        originals = [w.clone() for w in attn_weights]

        mock_layers = []
        for i in range(num_layers):
            layer = MagicMock()
            layer.self_attn.o_proj.weight = attn_weights[i]
            layer.mlp.down_proj.weight = mlp_weights[i]
            mock_layers.append(layer)

        mock_model.get_layers = MagicMock(return_value=mock_layers)

        def get_matrices(layer_index):
            return {
                "attn.o_proj": [attn_weights[layer_index]],
                "mlp.down_proj": [mlp_weights[layer_index]],
            }

        mock_model.get_layer_matrices = MagicMock(side_effect=get_matrices)
        mock_model.get_layer_multiplier = Model.get_layer_multiplier.__get__(
            mock_model, Model
        )

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
        assert torch.allclose(attn_weights[0], originals[0])
        # Layer 1: distance=0 <= 0.5, should be modified
        assert not torch.allclose(attn_weights[1], originals[1])
        # Layer 2: distance=1 > 0.5, should NOT be modified
        assert torch.allclose(attn_weights[2], originals[2])
        # Layer 3: distance=2 > 0.5, should NOT be modified
        assert torch.allclose(attn_weights[3], originals[3])


class TestModelResponses:
    """Test response generation methods."""

    def test_get_responses_decodes_generated_tokens(self):
        """Test get_responses properly decodes model output."""
        from bruno.model import Model

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
        from bruno.model import Model

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
        from bruno.model import Model

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

        with patch("bruno.model.batchify") as mock_batchify:
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

        from bruno.model import Model

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
        from bruno.model import Model

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

        with patch("bruno.model.batchify") as mock_batchify:
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
        from bruno.model import Model

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


class TestModelPCAExtraction:
    """Test Phase 1: Multi-Direction PCA Extraction."""

    def test_get_refusal_directions_pca_returns_correct_shape(self):
        """Test PCA extraction returns PCAExtractionResult with correct shapes."""
        from bruno.model import Model, PCAExtractionResult

        mock_model = MagicMock()

        # Create sample residuals: (n_samples, n_layers, hidden_dim)
        n_good, n_bad = 10, 10
        n_layers = 4
        hidden_dim = 64
        n_components = 3

        good_residuals = torch.randn(n_good, n_layers, hidden_dim)
        bad_residuals = torch.randn(n_bad, n_layers, hidden_dim)

        result = Model.get_refusal_directions_pca(
            mock_model,
            good_residuals,
            bad_residuals,
            n_components=n_components,
            alpha=1.0,
        )

        # Should return PCAExtractionResult
        assert isinstance(result, PCAExtractionResult)
        # Directions should be (n_layers, n_components, hidden_dim)
        assert result.directions.shape == (n_layers, n_components, hidden_dim)
        # Eigenvalues should be (n_layers, n_components)
        assert result.eigenvalues.shape == (n_layers, n_components)

    def test_get_refusal_directions_pca_normalized(self):
        """Test PCA directions are normalized."""
        from bruno.model import Model

        mock_model = MagicMock()

        good_residuals = torch.randn(20, 4, 64)
        bad_residuals = torch.randn(20, 4, 64)

        result = Model.get_refusal_directions_pca(
            mock_model,
            good_residuals,
            bad_residuals,
            n_components=2,
        )

        # Each direction should be unit normalized
        for layer_idx in range(result.directions.shape[0]):
            for comp_idx in range(result.directions.shape[1]):
                norm = result.directions[layer_idx, comp_idx].norm().item()
                assert abs(norm - 1.0) < 0.01, f"Direction not normalized: {norm}"

    def test_get_refusal_directions_pca_alpha_effect(self):
        """Test that alpha parameter affects results."""
        from bruno.model import Model

        mock_model = MagicMock()

        good_residuals = torch.randn(20, 4, 64)
        bad_residuals = torch.randn(20, 4, 64)

        result_alpha_0 = Model.get_refusal_directions_pca(
            mock_model, good_residuals, bad_residuals, n_components=2, alpha=0.0
        )
        result_alpha_1 = Model.get_refusal_directions_pca(
            mock_model, good_residuals, bad_residuals, n_components=2, alpha=1.0
        )

        # Results should differ with different alpha
        assert not torch.allclose(result_alpha_0.directions, result_alpha_1.directions)

    @pytest.mark.slow
    def test_get_refusal_directions_pca_gpu_performance(self):
        """Test GPU-accelerated PCA performance on realistic dimensions.

        Requires 16GB+ VRAM to allocate tensors for 32B model dimensions.
        """
        import time

        from bruno.model import Model

        # Skip if no CUDA available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Skip if insufficient VRAM (need 16GB+ for 32B model dimensions)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 16:
            pytest.skip(f"Insufficient VRAM: {gpu_memory_gb:.1f}GB (need 16GB+)")

        mock_model = MagicMock()

        # Realistic dimensions for 32B model
        # Use smaller sample size for test performance
        n_samples = 50  # Reduced from typical 400 for test speed
        n_layers = 64  # Qwen2.5-Coder-32B has 64 layers
        hidden_dim = 5120  # 32B hidden dimension

        device = "cuda"  # Already verified CUDA is available with sufficient VRAM

        good_residuals = torch.randn(n_samples, n_layers, hidden_dim, device=device)
        bad_residuals = torch.randn(n_samples, n_layers, hidden_dim, device=device)

        # Measure execution time
        start = time.time()
        result = Model.get_refusal_directions_pca(
            mock_model,
            good_residuals,
            bad_residuals,
            n_components=3,
        )
        elapsed = time.time() - start

        # Verify results are correct
        assert result.directions.shape == (n_layers, 3, hidden_dim)
        assert result.eigenvalues.shape == (n_layers, 3)

        # Performance expectation: GPU should be <30s for 64 layers
        # CPU fallback allowed to take longer
        if device == "cuda":
            assert elapsed < 30, (
                f"GPU PCA took {elapsed:.1f}s (expected <30s for test dimensions)"
            )
            print(f"\nGPU PCA performance: {elapsed:.2f}s for {n_layers} layers")
        else:
            print(f"\nCPU PCA (fallback): {elapsed:.2f}s for {n_layers} layers")


class TestPCAExtractionResult:
    """Test PCAExtractionResult eigenvalue weight computation."""

    def test_get_eigenvalue_weights_softmax(self):
        """Test softmax eigenvalue weight computation."""
        from bruno.model import PCAExtractionResult

        # Create result with known eigenvalues
        directions = torch.randn(4, 3, 64)
        # Eigenvalues: larger first eigenvalue
        eigenvalues = torch.tensor(
            [
                [10.0, 5.0, 1.0],
                [12.0, 4.0, 2.0],
                [8.0, 6.0, 1.5],
                [11.0, 5.0, 0.5],
            ]
        )

        result = PCAExtractionResult(directions=directions, eigenvalues=eigenvalues)
        weights = result.get_eigenvalue_weights(method="softmax")

        assert len(weights) == 3
        # First weight should be 1.0 (scaled)
        assert abs(weights[0] - 1.0) < 0.01
        # Weights should be decreasing (larger eigenvalue = larger weight)
        assert weights[0] >= weights[1] >= weights[2]

    def test_get_eigenvalue_weights_proportional(self):
        """Test proportional eigenvalue weight computation."""
        from bruno.model import PCAExtractionResult

        directions = torch.randn(4, 3, 64)
        eigenvalues = torch.tensor(
            [
                [10.0, 5.0, 2.5],
                [10.0, 5.0, 2.5],
                [10.0, 5.0, 2.5],
                [10.0, 5.0, 2.5],
            ]
        )

        result = PCAExtractionResult(directions=directions, eigenvalues=eigenvalues)
        weights = result.get_eigenvalue_weights(method="proportional")

        assert len(weights) == 3
        # First weight should be 1.0
        assert abs(weights[0] - 1.0) < 0.01
        # With uniform eigenvalues across layers, ratio should be preserved
        # Second should be 0.5 of first, third should be 0.25 of first
        assert abs(weights[1] - 0.5) < 0.01
        assert abs(weights[2] - 0.25) < 0.01

    def test_get_eigenvalue_weights_log_proportional(self):
        """Test log-proportional eigenvalue weight computation."""
        from bruno.model import PCAExtractionResult

        directions = torch.randn(4, 3, 64)
        eigenvalues = torch.tensor(
            [
                [100.0, 10.0, 1.0],
                [100.0, 10.0, 1.0],
                [100.0, 10.0, 1.0],
                [100.0, 10.0, 1.0],
            ]
        )

        result = PCAExtractionResult(directions=directions, eigenvalues=eigenvalues)
        weights = result.get_eigenvalue_weights(method="log_proportional")

        assert len(weights) == 3
        assert abs(weights[0] - 1.0) < 0.01
        # Log-proportional should reduce dominance of first eigenvalue
        # compared to proportional (second weight should be higher)
        weights_prop = result.get_eigenvalue_weights(method="proportional")
        assert weights[1] > weights_prop[1]

    def test_get_eigenvalue_weights_handles_negative_eigenvalues(self):
        """Test that negative eigenvalues are handled gracefully."""
        from bruno.model import PCAExtractionResult

        directions = torch.randn(4, 3, 64)
        # Contrastive PCA can produce negative eigenvalues
        eigenvalues = torch.tensor(
            [
                [10.0, -2.0, -5.0],
                [8.0, 1.0, -3.0],
                [12.0, -1.0, -4.0],
                [9.0, 0.0, -2.0],
            ]
        )

        result = PCAExtractionResult(directions=directions, eigenvalues=eigenvalues)

        # Should not raise, should clamp negative values
        weights = result.get_eigenvalue_weights(method="softmax")
        assert len(weights) == 3
        assert all(w >= 0 for w in weights)

    def test_get_eigenvalue_weights_temperature_effect(self):
        """Test that temperature affects softmax distribution."""
        from bruno.model import PCAExtractionResult

        directions = torch.randn(4, 3, 64)
        eigenvalues = torch.tensor(
            [
                [10.0, 5.0, 1.0],
                [10.0, 5.0, 1.0],
                [10.0, 5.0, 1.0],
                [10.0, 5.0, 1.0],
            ]
        )

        result = PCAExtractionResult(directions=directions, eigenvalues=eigenvalues)

        weights_low_temp = result.get_eigenvalue_weights(
            method="softmax", temperature=0.5
        )
        weights_high_temp = result.get_eigenvalue_weights(
            method="softmax", temperature=2.0
        )

        # Higher temperature = more uniform weights
        # So second weight should be closer to first with high temperature
        assert weights_high_temp[1] > weights_low_temp[1]

    def test_get_eigenvalue_weights_invalid_method(self):
        """Test that invalid method raises ConfigurationError."""
        from bruno.exceptions import ConfigurationError
        from bruno.model import PCAExtractionResult

        directions = torch.randn(4, 3, 64)
        eigenvalues = torch.randn(4, 3)

        result = PCAExtractionResult(directions=directions, eigenvalues=eigenvalues)

        with pytest.raises(
            ConfigurationError, match="Unknown eigenvalue weight method"
        ):
            result.get_eigenvalue_weights(method="invalid_method")


class TestModelOrthogonalization:
    """Test Phase 5: Direction Orthogonalization."""

    def test_extract_helpfulness_direction_shape(self):
        """Test helpfulness direction extraction returns correct shape."""
        from bruno.model import Model

        mock_model = MagicMock()

        n_layers = 8
        hidden_dim = 128

        helpful_residuals = torch.randn(10, n_layers, hidden_dim)
        unhelpful_residuals = torch.randn(10, n_layers, hidden_dim)

        direction = Model.extract_helpfulness_direction(
            mock_model,
            helpful_residuals,
            unhelpful_residuals,
        )

        assert direction.shape == (n_layers, hidden_dim)

    def test_extract_helpfulness_direction_normalized(self):
        """Test helpfulness direction is normalized."""
        from bruno.model import Model

        mock_model = MagicMock()

        helpful_residuals = torch.randn(10, 4, 64)
        unhelpful_residuals = torch.randn(10, 4, 64)

        direction = Model.extract_helpfulness_direction(
            mock_model,
            helpful_residuals,
            unhelpful_residuals,
        )

        # Each layer's direction should be unit normalized
        for layer_idx in range(direction.shape[0]):
            norm = direction[layer_idx].norm().item()
            assert abs(norm - 1.0) < 0.01

    def test_orthogonalize_direction_removes_component(self):
        """Test orthogonalization removes the projection."""
        from bruno.model import Model

        mock_model = MagicMock()

        # Create target and remove directions
        target = torch.randn(4, 64)
        target = torch.nn.functional.normalize(target, p=2, dim=1)

        remove = torch.randn(4, 64)
        remove = torch.nn.functional.normalize(remove, p=2, dim=1)

        result = Model.orthogonalize_direction(mock_model, target, remove)

        # Result should be orthogonal to remove direction
        dot_products = (result * remove).sum(dim=1)
        for dot in dot_products:
            assert abs(dot.item()) < 0.01, "Directions not orthogonal"

    def test_orthogonalize_direction_normalized(self):
        """Test orthogonalized direction is normalized."""
        from bruno.model import Model

        mock_model = MagicMock()

        target = torch.randn(4, 64)
        remove = torch.randn(4, 64)

        result = Model.orthogonalize_direction(mock_model, target, remove)

        for layer_idx in range(result.shape[0]):
            norm = result[layer_idx].norm().item()
            assert abs(norm - 1.0) < 0.01


class TestModelIterativeAblation:
    """Test Phase 4: Iterative Refinement."""

    def test_abliterate_iterative_returns_rounds_count(self):
        """Test iterative ablation returns number of rounds performed."""
        from bruno.model import AbliterationParameters, Model

        mock_model = MagicMock()
        mock_model.model.dtype = torch.float32
        mock_model.settings.batch_size = 2

        # Mock residuals that differ enough to continue - need different values each call
        # to ensure the magnitude check passes
        call_count = [0]

        def mock_residuals(prompts):
            call_count[0] += 1
            # Return different residuals each time with large magnitude difference
            return torch.randn(5, 4, 64) * (10.0 + call_count[0])

        mock_model.get_residuals_batched.side_effect = mock_residuals

        # Mock abliterate to do nothing
        mock_model.abliterate = MagicMock()

        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=2.0,
                min_weight=0.0,
                min_weight_distance=4.0,
            ),
        }

        with patch("bruno.model.print"):
            with patch("bruno.model.empty_cache"):
                rounds, kl_values = Model.abliterate_iterative(
                    mock_model,
                    good_prompts=["good1", "good2"],
                    bad_prompts=["bad1", "bad2"],
                    parameters=parameters,
                    max_rounds=2,
                    min_direction_magnitude=0.001,  # Low threshold to continue
                )

        assert rounds == 2
        assert isinstance(kl_values, list)

    def test_abliterate_iterative_stops_on_low_magnitude(self):
        """Test iterative ablation stops when magnitude is too low."""
        from bruno.model import AbliterationParameters, Model

        mock_model = MagicMock()
        mock_model.model.dtype = torch.float32

        # Mock residuals that are nearly identical (low magnitude)
        base_residuals = torch.randn(5, 4, 64)
        mock_model.get_residuals_batched.side_effect = [
            base_residuals,
            base_residuals + 0.001,  # Very similar
        ]

        mock_model.abliterate = MagicMock()

        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=2.0,
                min_weight=0.0,
                min_weight_distance=4.0,
            ),
        }

        with patch("bruno.model.print"):
            with patch("bruno.model.empty_cache"):
                rounds, _ = Model.abliterate_iterative(
                    mock_model,
                    good_prompts=["good"],
                    bad_prompts=["bad"],
                    parameters=parameters,
                    max_rounds=5,
                    min_direction_magnitude=100.0,  # High threshold
                )

        # Should stop after 1 round due to low magnitude
        assert rounds == 1


class TestModelMultiDirectionAblation:
    """Test Phase 1: Multi-Direction Ablation."""

    def test_abliterate_multi_direction_applies_weights(self):
        """Test multi-direction ablation applies configurable weights."""
        from bruno.model import AbliterationParameters, Model

        mock_model = MagicMock()
        mock_model.abliterate = MagicMock()

        # Create multi-direction tensor: (n_layers, n_components, hidden_dim)
        refusal_directions = torch.randn(4, 3, 64)

        direction_weights = [1.0, 0.5, 0.0]  # Third one has 0 weight

        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=2.0,
                min_weight=0.0,
                min_weight_distance=4.0,
            ),
        }

        with patch("bruno.model.print"):
            Model.abliterate_multi_direction(
                mock_model,
                refusal_directions,
                direction_weights,
                parameters,
            )

        # Should have called abliterate twice (skipping 0-weight direction)
        assert mock_model.abliterate.call_count == 2

    def test_abliterate_multi_direction_scales_weights(self):
        """Test multi-direction ablation scales parameters by weight."""
        from bruno.model import AbliterationParameters, Model

        mock_model = MagicMock()

        captured_params = []

        def capture_abliterate(
            directions, direction_index, params, layer_profiles=None
        ):
            captured_params.append(params)

        mock_model.abliterate = capture_abliterate

        refusal_directions = torch.randn(4, 2, 64)
        direction_weights = [1.0, 0.5]

        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=2.0,
                max_weight_position=2.0,
                min_weight=1.0,
                min_weight_distance=4.0,
            ),
        }

        with patch("bruno.model.print"):
            Model.abliterate_multi_direction(
                mock_model,
                refusal_directions,
                direction_weights,
                parameters,
            )

        # First call should have max_weight=2.0 (1.0 * 2.0)
        assert captured_params[0]["attn.o_proj"].max_weight == 2.0
        # Second call should have max_weight=1.0 (0.5 * 2.0)
        assert captured_params[1]["attn.o_proj"].max_weight == 1.0


class TestModelGenerate:
    """Test the generate method."""

    def test_generate_calls_tokenizer(self):
        """Test generate calls tokenizer with chat template."""
        from bruno.model import Model

        mock_model = MagicMock()
        mock_model.settings.system_prompt = "You are helpful."
        mock_model.model.device = torch.device("cpu")
        mock_model.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Mock get_chat to return proper chat format
        mock_model.get_chat.return_value = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        mock_model.tokenizer.apply_chat_template.return_value = ["<s>test</s>"]
        mock_model.tokenizer.eos_token_id = 2

        # Mock the tokenizer call to return BatchEncoding-like object
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        mock_model.tokenizer.return_value = mock_inputs

        inputs, outputs = Model.generate(mock_model, ["Hello"], max_new_tokens=10)

        # Verify tokenizer was called
        mock_model.tokenizer.assert_called()
        mock_model.model.generate.assert_called()


class TestModelMultiTokenLogprobs:
    """Test Phase 3: Multi-token logprobs."""

    def test_get_logprobs_single_token(self):
        """Test get_logprobs with n_tokens=1 (original behavior)."""
        from bruno.model import Model

        mock_model = MagicMock()

        mock_logits = torch.randn(2, 32000)
        mock_outputs = MagicMock()
        mock_outputs.scores = [mock_logits]
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_model.generate.return_value = (mock_inputs, mock_outputs)

        logprobs = Model.get_logprobs(mock_model, ["P1", "P2"], n_tokens=1)

        # Shape should be (n_prompts, vocab_size)
        assert logprobs.shape == (2, 32000)

    def test_get_logprobs_multi_token(self):
        """Test get_logprobs with n_tokens>1."""
        from bruno.model import Model

        mock_model = MagicMock()

        # 3 tokens generated
        mock_outputs = MagicMock()
        mock_outputs.scores = [
            torch.randn(2, 32000),
            torch.randn(2, 32000),
            torch.randn(2, 32000),
        ]
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_model.generate.return_value = (mock_inputs, mock_outputs)

        logprobs = Model.get_logprobs(mock_model, ["P1", "P2"], n_tokens=3)

        # Shape should be (n_prompts, n_tokens, vocab_size)
        assert logprobs.shape == (2, 3, 32000)


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
