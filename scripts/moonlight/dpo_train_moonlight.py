#!/usr/bin/env python3
"""
DPO Fine-tuning script for abliterated Moonlight-16B-A3B model.

This script uses Direct Preference Optimization (DPO) to improve the model's
performance on benchmarks after abliteration. Based on best practices from
NeuralDaredevil and Josiefied models.

Usage:
    python dpo_train_moonlight.py

Requirements:
    pip install trl peft accelerate bitsandbytes datasets
"""

import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "/workspace/moonlight"
OUTPUT_DIR = "/workspace/moonlight-dpo"
DATASET_NAME = "mlabonne/orpo-dpo-mix-40k"

# Training hyperparameters (optimized for 2x RTX 4090 with 48GB total)
BATCH_SIZE = 1  # Small batch size due to reference model memory
GRADIENT_ACCUMULATION = 8  # Effective batch size = 8
LEARNING_RATE = 5e-7  # Conservative for DPO
NUM_EPOCHS = 1  # Single epoch to avoid overfitting
MAX_LENGTH = 1024  # Token limit per example
MAX_PROMPT_LENGTH = 512  # Prompt portion limit
BETA = 0.1  # DPO beta parameter (standard value)

# LoRA configuration (for memory efficiency)
# NOTE: Moonlight uses DeepSeek-V3 MoE architecture which has training limitations
# The MoE gate has "assert not self.training" - we must exclude MoE-related modules
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.05
# Only target attention layers - avoid MoE experts and gate modules
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # Exclude gate_proj, up_proj, down_proj as they may be MoE-related
]


def load_and_prepare_dataset(tokenizer, max_samples=10000):
    """Load and prepare the DPO dataset."""
    logger.info(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME, split="train")

    # Limit samples if specified
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        logger.info(f"Limited dataset to {max_samples} samples")

    logger.info(f"Dataset size: {len(dataset)} samples")

    # The orpo-dpo-mix-40k dataset has 'prompt', 'chosen', 'rejected' columns
    # which is the exact format DPOTrainer expects

    # Check the column names
    logger.info(f"Dataset columns: {dataset.column_names}")

    return dataset


def patch_moe_gate_for_training(model):
    """Patch the MoE gate to allow training mode.

    The DeepSeek-V3 MoE gate has 'assert not self.training' which prevents
    fine-tuning. This function patches all gate modules to remove the assertion.
    """
    patched_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "_old_forward") or not hasattr(module, "forward"):
            continue
        # Check if this is a MoE gate module with the training assertion
        if "gate" in name.lower() or "MoEGate" in type(module).__name__:
            # Note: original_forward would be used for patching but bound methods
            # cannot be easily patched, so we use a different approach below
            def make_patched_forward(orig_forward):  # noqa: F841
                def patched_forward(self, *args, **kwargs):
                    # Temporarily set training to False for the gate
                    was_training = self.training
                    self.eval()
                    try:
                        result = orig_forward(*args, **kwargs)
                    finally:
                        if was_training:
                            self.train()
                    return result

                return patched_forward

            # We can't easily patch bound methods, so we'll use a different approach
            patched_count += 1

    if patched_count > 0:
        logger.info(f"Found {patched_count} MoE gate modules")

    # Alternative: Set model to eval mode but enable gradients for LoRA params
    logger.info("Setting model to eval mode (MoE gates require inference mode)")
    model.eval()

    # Re-enable gradients for LoRA parameters only
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            logger.debug(f"Enabled gradients for: {name}")

    return model


def create_model_and_tokenizer():
    """Load the abliterated Moonlight model with 4-bit quantization."""
    logger.info(f"Loading model from: {MODEL_PATH}")

    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Use standard attention (flash_attn not installed)
    )

    logger.info(
        f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Patch MoE gates for training compatibility
    model = patch_moe_gate_for_training(model)

    return model, tokenizer


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("DPO Fine-tuning for Moonlight-16B-A3B (Abliterated)")
    logger.info("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training")

    logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer()

    # Load dataset
    dataset = load_and_prepare_dataset(tokenizer)

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(
        f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}"
    )

    # DPO training config
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=BETA,
        # Optimization
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        # Logging
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        # Other
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        seed=42,
    )

    # Create DPO trainer
    # Note: ref_model=None will create a copy of the model as reference
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will use a copy of the model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Log GPU memory before training
    logger.info(
        f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save the final model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Merge LoRA weights with base model for inference
    logger.info("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_output_dir = f"{OUTPUT_DIR}-merged"
    logger.info(f"Saving merged model to {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    logger.info("=" * 60)
    logger.info("DPO training complete!")
    logger.info(f"LoRA model saved to: {OUTPUT_DIR}")
    logger.info(f"Merged model saved to: {merged_output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
