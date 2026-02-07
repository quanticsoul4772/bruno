# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Extended benchmark evaluators for model capability assessment.

Provides evaluators for popular benchmarks beyond MMLU:
- HellaSwag: Common-sense reasoning
- ARC-Challenge: Advanced reasoning
- TruthfulQA: Factual accuracy
- GSM8K: Mathematical reasoning
- HumanEval: Code generation (for coding models)
"""

import re
from typing import TYPE_CHECKING

from datasets import load_dataset

from .logging import get_logger

if TYPE_CHECKING:
    from .model import Model

logger = get_logger(__name__)


class HellaSwagEvaluator:
    """Evaluator for HellaSwag common-sense reasoning benchmark."""

    def __init__(self, model: "Model", num_samples: int = 100):
        """Initialize HellaSwag evaluator.

        Args:
            model: The model to evaluate
            num_samples: Number of samples to evaluate (default: 100)
        """
        self.model = model
        self.num_samples = num_samples
        self.name = "HellaSwag"

    def evaluate(self) -> float:
        """Evaluate model on HellaSwag.

        Returns:
            Accuracy as a float between 0 and 1
        """
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        correct = 0
        total = 0

        for item in dataset:
            # HellaSwag format: context + 4 endings
            context = item["ctx"]
            endings = item["endings"]
            correct_idx = int(item["label"])

            # Create prompts for each ending
            prompts = [
                f"{context} {ending}\n\nIs this a logical completion? (yes/no)"
                for ending in endings
            ]

            # Get model scores for each ending
            scores = []
            for prompt in prompts:
                # Simple scoring: check if model says "yes"
                inputs = self.model.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )
                outputs = self.model.model.generate(
                    **inputs, max_new_tokens=5, do_sample=False
                )
                response = self.model.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                # Score based on "yes" confidence
                score = 1.0 if "yes" in response.lower() else 0.0
                scores.append(score)

            # Select highest scoring ending
            predicted_idx = scores.index(max(scores))

            if predicted_idx == correct_idx:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0


class ARCEvaluator:
    """Evaluator for ARC-Challenge reasoning benchmark."""

    def __init__(self, model: "Model", num_samples: int = 100):
        """Initialize ARC evaluator.

        Args:
            model: The model to evaluate
            num_samples: Number of samples to evaluate
        """
        self.model = model
        self.num_samples = num_samples
        self.name = "ARC-Challenge"

    def evaluate(self) -> float:
        """Evaluate model on ARC-Challenge.

        Returns:
            Accuracy as a float between 0 and 1
        """
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        correct = 0
        total = 0

        for item in dataset:
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]

            # Find correct index
            correct_idx = labels.index(answer_key)

            # Create multiple choice prompt
            prompt = f"{question}\n\n"
            for i, (label, choice) in enumerate(zip(labels, choices)):
                prompt += f"{label}. {choice}\n"
            prompt += "\nAnswer (A/B/C/D):"

            # Get model response
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model.model.generate(
                **inputs, max_new_tokens=5, do_sample=False
            )
            response = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer letter
            match = re.search(r"\b([A-D])\b", response.split("Answer")[-1])
            if match:
                predicted_label = match.group(1)
                predicted_idx = (
                    labels.index(predicted_label) if predicted_label in labels else -1
                )

                if predicted_idx == correct_idx:
                    correct += 1

            total += 1

        return correct / total if total > 0 else 0.0


class GSM8KEvaluator:
    """Evaluator for GSM8K mathematical reasoning benchmark."""

    def __init__(self, model: "Model", num_samples: int = 100):
        """Initialize GSM8K evaluator.

        Args:
            model: The model to evaluate
            num_samples: Number of samples to evaluate
        """
        self.model = model
        self.num_samples = num_samples
        self.name = "GSM8K"

    def evaluate(self) -> float:
        """Evaluate model on GSM8K.

        Returns:
            Accuracy as a float between 0 and 1
        """
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        correct = 0
        total = 0

        for item in dataset:
            question = item["question"]
            answer = item["answer"]

            # Extract numerical answer
            answer_match = re.search(r"#### (.+)", answer)
            if not answer_match:
                continue
            expected_answer = answer_match.group(1).strip()

            # Prompt with chain-of-thought
            prompt = f"{question}\n\nLet's solve this step by step:"

            # Get model response
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
            response = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract final answer (look for numbers in the response)
            numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response)
            if numbers:
                predicted_answer = numbers[-1].replace(",", "")  # Last number
                if predicted_answer == expected_answer.replace(",", ""):
                    correct += 1

            total += 1

        return correct / total if total > 0 else 0.0


class BenchmarkSuite:
    """Orchestrator for running multiple benchmarks."""

    def __init__(self, model: "Model", benchmark_suites: list[str], samples: int = 100):
        """Initialize benchmark suite.

        Args:
            model: The model to evaluate
            benchmark_suites: List of benchmark names to run
            samples: Number of samples per benchmark
        """
        self.model = model
        self.evaluators = []

        for name in benchmark_suites:
            if name == "hellaswag":
                self.evaluators.append(HellaSwagEvaluator(model, samples))
            elif name == "arc_challenge":
                self.evaluators.append(ARCEvaluator(model, samples))
            elif name == "gsm8k":
                self.evaluators.append(GSM8KEvaluator(model, samples))
            else:
                logger.warning(f"Unknown benchmark: {name}")

    def run_all(self) -> dict[str, float]:
        """Run all configured benchmarks.

        Returns:
            Dictionary mapping benchmark name to accuracy score
        """
        results = {}
        for evaluator in self.evaluators:
            logger.info(f"Running {evaluator.name} benchmark...")
            try:
                score = evaluator.evaluate()
                results[evaluator.name] = score
                logger.info(f"{evaluator.name}: {score:.1%}")
            except Exception as e:
                logger.error(f"{evaluator.name} failed: {e}")
                results[evaluator.name] = 0.0

        return results
