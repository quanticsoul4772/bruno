# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Interactive chat interface for abliterated models.

Provides a terminal-based chat experience similar to Claude Code
for testing and using abliterated models.
"""

import os
import sys
from typing import Optional

# Disable symlinks on Windows to avoid permission issues
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
from huggingface_hub import list_models
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

from .logging import get_logger
from .utils import empty_cache
from .utils import print as rich_print

logger = get_logger(__name__)


def select_model_interactive(default: str = "rawcell/bruno") -> str:
    """Interactively select a model from HuggingFace.

    Args:
        default: Default model to pre-select

    Returns:
        Selected model name
    """
    import questionary

    rich_print("[bold]Select a model to chat with:[/]")
    rich_print()

    # Get user's models from HuggingFace
    try:
        user_models = list(list_models(author="rawcell", sort="lastModified", limit=20))
        model_names = [m.id for m in user_models if m.id]

        if not model_names:
            rich_print("[yellow]No models found in your HuggingFace account.[/]")
            return default

        # Add popular alternatives
        popular = [
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]

        all_choices = model_names + ["---"] + popular + ["---", "Enter custom model"]

        # Select using questionary
        selected = questionary.select(
            "Choose model:",
            choices=all_choices,
            default=default if default in all_choices else all_choices[0],
        ).ask()

        if selected == "Enter custom model":
            selected = questionary.text(
                "Enter model name (HuggingFace ID or local path):"
            ).ask()

        if selected == "---":
            return default

        return selected

    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
        rich_print(f"[yellow]Using default: {default}[/]")
        return default


class BrunoChat:
    """Interactive chat session with an abliterated model.

    Provides streaming responses, conversation history, and
    a clean terminal interface for multi-turn dialogues.
    """

    def __init__(
        self,
        model_name: str,
        use_4bit: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """Initialize chat session.

        Args:
            model_name: HuggingFace model ID or local path
            use_4bit: Use 4-bit quantization (for 8GB VRAM)
            temperature: Sampling temperature (0.0-1.0)
            max_new_tokens: Max tokens per response
            system_prompt: Optional system prompt to prepend
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.history: list[dict] = []

        # Add system prompt if provided
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

        rich_print(f"Loading [bold]{model_name}[/]...")

        # Configure quantization
        quantization_config = None
        if use_4bit:
            rich_print("  * Using 4-bit quantization for RTX 4070 (8GB VRAM)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load model (pass token if available)
        token = os.environ.get("HF_TOKEN")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto" if not use_4bit else None,
            token=token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

        # Setup streaming output
        self.streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        rich_print("[green]✓[/] Bruno ready!")
        rich_print()
        rich_print("[dim]Commands:[/]")
        rich_print("  [bold]/clear[/]  - Clear conversation history")
        rich_print("  [bold]/exit[/]   - Exit chat")
        rich_print("  [bold]Ctrl+C[/]  - Exit chat")
        rich_print()

    def chat(self, user_message: str) -> str:
        """Send message and get streaming response.

        Args:
            user_message: User's message

        Returns:
            Model's complete response text
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Apply chat template
        try:
            inputs = self.tokenizer.apply_chat_template(
                self.history,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}. Using simple concatenation.")
            # Fallback for models without chat template
            messages_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.history]
            )
            inputs = self.tokenizer(messages_text, return_tensors="pt")["input_ids"]

        # Extract input_ids tensor if inputs is a dict/BatchEncoding
        if isinstance(inputs, dict):
            inputs = inputs["input_ids"]
        elif hasattr(inputs, "input_ids"):
            inputs = inputs.input_ids

        inputs = inputs.to(self.model.device)

        # Generate with streaming
        rich_print("[bold cyan]Bruno:[/] ", end="")

        try:
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50,
                streamer=self.streamer,  # Streams to terminal in real-time
            )

            # Extract response text
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1] :],
                skip_special_tokens=True,
            )

            # Add to history
            self.history.append({"role": "assistant", "content": response})

            return response

        except torch.cuda.OutOfMemoryError as e:
            rich_print(f"\n[red]GPU out of memory: {e}[/]")
            rich_print(f"[yellow]Try: bruno chat --model {self.model_name} --4bit[/]")
            empty_cache()
            return "[ERROR: OOM]"

        except Exception as e:
            # Show full error with traceback for debugging
            import traceback

            logger.error(f"Generation error: {e}")
            rich_print("\n[red]Error generating response:[/]")
            rich_print(f"[red]{e}[/]")
            rich_print("\n[dim]Full traceback:[/]")
            traceback.print_exc()
            return f"[ERROR: {e}]"

    def run_interactive(self):
        """Run interactive chat loop with command handling."""
        while True:
            try:
                # Get user input
                rich_print()
                user_input = input("[bold green]You:[/] ")

                if not user_input.strip():
                    continue

                # Handle commands
                command = user_input.strip().lower()

                if command in ["/exit", "/quit", "/q"]:
                    rich_print("[dim]Goodbye![/]")
                    break

                if command == "/clear":
                    # Keep system prompt if exists
                    system_msg = next(
                        (m for m in self.history if m["role"] == "system"), None
                    )
                    self.history = [system_msg] if system_msg else []
                    empty_cache()
                    rich_print("[dim]History cleared.[/]")
                    continue

                if command == "/help":
                    rich_print("[bold]Available commands:[/]")
                    rich_print("  [bold]/clear[/]  - Clear conversation history")
                    rich_print("  [bold]/exit[/]   - Exit chat")
                    rich_print("  [bold]/help[/]   - Show this help")
                    rich_print()
                    rich_print(f"[dim]Model: {self.model_name}[/]")
                    rich_print(
                        f"[dim]History: {len([m for m in self.history if m['role'] != 'system'])} messages[/]"
                    )
                    continue

                # Chat
                self.chat(user_input)

            except KeyboardInterrupt:
                rich_print()
                rich_print("[dim]Goodbye![/]")
                break

            except Exception as e:
                logger.error(f"Chat loop error: {e}")
                rich_print(f"\n[red]Error: {e}[/]")
                rich_print("[yellow]Continuing...[/]")


def run_chat(
    model_name: Optional[str] = None,
    use_4bit: bool = False,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    system_prompt: Optional[str] = None,
):
    """Entry point for chat mode.

    Args:
        model_name: HuggingFace model ID or local path (None = interactive selector)
        use_4bit: Use 4-bit quantization
        temperature: Sampling temperature
        max_new_tokens: Max tokens per response
        system_prompt: Optional system prompt
    """
    # Print banner
    rich_print()
    rich_print("[cyan]█▀▄░█▀▄░█░█░█▄░█░█▀█[/]  Chat Mode")
    rich_print("[cyan]█▀▄░█▀▄░█░█░█░▀█░█░█[/]")
    rich_print("[cyan]▀▀░░▀░▀░▀▀▀░▀░░▀░▀▀▀[/]  Interactive chat with your model")
    rich_print()

    # Interactive model selection if not specified
    if model_name is None:
        model_name = select_model_interactive(default="rawcell/bruno")
        rich_print()

    try:
        # Create and run chat session
        chat_session = BrunoChat(
            model_name=model_name,
            use_4bit=use_4bit,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )

        chat_session.run_interactive()

    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        rich_print(f"[red]Failed to start chat: {e}[/]")
        sys.exit(1)
