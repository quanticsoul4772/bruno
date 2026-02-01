# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Command-line interface routing for Bruno.

Provides subcommands:
- abliterate: Run abliteration pipeline (default)
- chat: Interactive chat with abliterated models
"""

import os
import sys

# Set environment variables before any HuggingFace imports
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set HF_TOKEN from .env if not already set
if "HF_TOKEN" not in os.environ:
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["HF_TOKEN"] = token
                    break

import click


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Bruno - Neural behavior engineering framework.

    Named after Giordano Bruno (1548-1600) who revealed infinite
    cosmic worlds against imposed constraints.

    \b
    Commands:
      abliterate    Run abliteration on a model (default)
      chat          Interactive chat with your model

    \b
    Examples:
      bruno --model Qwen/Qwen2.5-7B-Instruct --n-trials 200
      bruno abliterate --model Qwen/Qwen2.5-7B
      bruno chat --model rawcell/bruno --4bit
    """
    # If no subcommand provided, run abliterate as default
    if ctx.invoked_subcommand is None:
        ctx.invoke(abliterate)


@cli.command()
def abliterate():
    """Run abliteration on a model (default command).

    This is the main Bruno functionality - modifying model behaviors
    through activation direction analysis and Optuna optimization.

    All flags are passed through Pydantic Settings (parsed from sys.argv).

    \b
    Example:
      bruno abliterate --model Qwen/Qwen2.5-7B-Instruct --n-trials 200
      bruno --model Qwen/Qwen2.5-7B  # (abliterate is default)
    """
    from .main import main as run_abliteration

    run_abliteration()


@cli.command()
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to chat with (HuggingFace ID or local path). If not specified, shows interactive selector.",
)
@click.option(
    "--4bit",
    "use_4bit",
    is_flag=True,
    help="Use 4-bit quantization (for 8GB VRAM like RTX 4070)",
)
@click.option(
    "--temperature",
    "-t",
    default=0.7,
    type=float,
    help="Sampling temperature (0.0-1.0)",
    show_default=True,
)
@click.option(
    "--max-tokens",
    default=2048,
    type=int,
    help="Maximum tokens per response",
    show_default=True,
)
@click.option(
    "--system-prompt",
    "-s",
    default=None,
    help="System prompt to prepend to conversation",
)
def chat(model, use_4bit, temperature, max_tokens, system_prompt):
    """Interactive chat with your abliterated model.

    Launch an interactive terminal chat session with streaming responses.
    Maintains conversation history for multi-turn dialogues.

    \b
    Commands during chat:
      /clear    Clear conversation history
      /exit     Exit chat
      Ctrl+C    Exit chat

    \b
    Examples:
      bruno chat
      bruno chat --model rawcell/bruno --4bit
      bruno chat -m ./models/my-abliterated-model --temperature 0.9
      bruno chat --system-prompt "You are a coding assistant"
    """
    try:
        from .chat import run_chat
    except ImportError as e:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Failed to import chat module: {e}"
        )
        click.echo(
            click.style("Hint: ", fg="yellow")
            + "Make sure transformers and torch are installed"
        )
        sys.exit(1)

    try:
        run_chat(
            model_name=model,
            use_4bit=use_4bit,
            temperature=temperature,
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    except KeyboardInterrupt:
        click.echo()  # New line after Ctrl+C
        click.echo(click.style("Goodbye!", fg="cyan"))
    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True))
        sys.exit(1)


if __name__ == "__main__":
    cli()
