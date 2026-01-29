# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Base class for direction extractors."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from heretic.config import Settings
    from heretic.model import Model


class DirectionExtractor(ABC):
    """Abstract base class for behavioral direction extractors.
    
    Direction extractors compute the "behavioral direction" in activation space
    that corresponds to a particular behavior (e.g., refusal, verbosity).
    
    To create a custom extractor:
    
    1. Subclass DirectionExtractor
    2. Implement get_prompts() to return contrastive prompt pairs
    3. Optionally override extract_directions() for custom logic
    4. Register with: register_extractor("my_behavior", MyExtractor)
    
    Example:
        class VerbosityExtractor(DirectionExtractor):
            def get_prompts(self, settings):
                return (
                    ["Be concise.", "Short answer."],  # positive (concise)
                    ["Explain in detail.", "Be verbose."],  # negative (verbose)
                )
    """
    
    @abstractmethod
    def get_prompts(self, settings: "Settings") -> tuple[list[str], list[str]]:
        """Get contrastive prompt pairs for direction extraction.
        
        Args:
            settings: Heretic settings (contains dataset specifications)
            
        Returns:
            Tuple of (positive_prompts, negative_prompts) where:
            - positive_prompts: Prompts that elicit the desired behavior
            - negative_prompts: Prompts that elicit the opposite behavior
            
            The direction is computed as: mean(negative) - mean(positive)
        """
        pass
    
    def extract_directions(
        self,
        model: "Model",
        positive_prompts: list[str],
        negative_prompts: list[str],
    ) -> Tensor:
        """Extract per-layer behavioral directions.
        
        Default implementation computes normalized difference of mean residuals.
        Override this method for custom extraction logic.
        
        Args:
            model: Loaded heretic Model instance
            positive_prompts: Prompts eliciting desired behavior
            negative_prompts: Prompts eliciting opposite behavior
            
        Returns:
            Tensor of shape (num_layers+1, hidden_dim) containing
            normalized direction vectors for embeddings + each layer
        """
        import torch.nn.functional as F
        
        # Get residuals for both prompt sets
        positive_residuals = model.get_residuals_batched(positive_prompts)
        negative_residuals = model.get_residuals_batched(negative_prompts)
        
        # Compute normalized direction: negative - positive
        directions = F.normalize(
            negative_residuals.mean(dim=0) - positive_residuals.mean(dim=0),
            p=2,
            dim=1,
        )
        
        return directions
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
