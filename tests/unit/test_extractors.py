# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Unit tests for direction extractor plugin system."""

import pytest
from unittest.mock import MagicMock, patch


class TestExtractorRegistry:
    """Test extractor registration and lookup."""
    
    def test_list_extractors_includes_refusal(self):
        """Test refusal extractor is registered by default."""
        from heretic.extractors import list_extractors
        
        extractors = list_extractors()
        
        assert "refusal" in extractors
    
    def test_get_extractor_returns_class(self):
        """Test get_extractor returns the extractor class."""
        from heretic.extractors import get_extractor, RefusalDirectionExtractor
        
        extractor_cls = get_extractor("refusal")
        
        assert extractor_cls is RefusalDirectionExtractor
    
    def test_get_extractor_unknown_raises(self):
        """Test get_extractor raises KeyError for unknown name."""
        from heretic.extractors import get_extractor
        
        with pytest.raises(KeyError, match="Unknown extractor.*nonexistent"):
            get_extractor("nonexistent")
    
    def test_register_extractor(self):
        """Test registering a custom extractor."""
        from heretic.extractors import register_extractor, get_extractor, DirectionExtractor
        
        class CustomExtractor(DirectionExtractor):
            def get_prompts(self, settings):
                return ["a"], ["b"]
        
        register_extractor("custom_test", CustomExtractor)
        
        assert get_extractor("custom_test") is CustomExtractor
    
    def test_register_extractor_validates_type(self):
        """Test register_extractor rejects non-DirectionExtractor classes."""
        from heretic.extractors import register_extractor
        
        class NotAnExtractor:
            pass
        
        with pytest.raises(TypeError, match="must inherit from DirectionExtractor"):
            register_extractor("invalid", NotAnExtractor)


class TestDirectionExtractor:
    """Test DirectionExtractor base class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test DirectionExtractor cannot be instantiated directly."""
        from heretic.extractors import DirectionExtractor
        
        with pytest.raises(TypeError, match="abstract"):
            DirectionExtractor()
    
    def test_subclass_must_implement_get_prompts(self):
        """Test subclass must implement get_prompts."""
        from heretic.extractors import DirectionExtractor
        
        class IncompleteExtractor(DirectionExtractor):
            pass
        
        with pytest.raises(TypeError):
            IncompleteExtractor()
    
    def test_extract_directions_default_impl(self):
        """Test default extract_directions implementation."""
        from heretic.extractors import DirectionExtractor
        import torch
        
        class TestExtractor(DirectionExtractor):
            def get_prompts(self, settings):
                return ["good"], ["bad"]
        
        extractor = TestExtractor()
        
        # Mock model - residuals shape is (batch, layers, hidden_dim)
        mock_model = MagicMock()
        # Return different residuals for positive and negative prompts
        mock_model.get_residuals_batched.side_effect = [
            torch.randn(1, 32, 4096),  # positive prompts
            torch.randn(1, 32, 4096),  # negative prompts
        ]
        
        directions = extractor.extract_directions(
            mock_model, ["good"], ["bad"]
        )
        
        # Should return normalized directions - shape is (layers, hidden_dim)
        assert directions.shape == (32, 4096)
        # Should be normalized (L2 norm ~= 1)
        norms = torch.norm(directions, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.01)


class TestRefusalDirectionExtractor:
    """Test refusal direction extractor."""
    
    def test_get_prompts_loads_from_settings(self):
        """Test get_prompts loads datasets from settings."""
        from heretic.extractors import RefusalDirectionExtractor
        
        extractor = RefusalDirectionExtractor()
        
        # Mock settings with DatasetSpecification
        mock_settings = MagicMock()
        
        with patch("heretic.utils.load_prompts") as mock_load:
            mock_load.side_effect = [
                ["harmless prompt 1", "harmless prompt 2"],
                ["harmful prompt 1", "harmful prompt 2"],
            ]
            
            positive, negative = extractor.get_prompts(mock_settings)
            
            assert len(positive) == 2
            assert len(negative) == 2
            assert mock_load.call_count == 2
