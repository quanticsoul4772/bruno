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
        try:
            # Try different encodings for Windows
            for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    with open(env_file, encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("HF_TOKEN="):
                                token = (
                                    line.split("=", 1)[1].strip().strip('"').strip("'")
                                )
                                if token and token != "your_token_here":
                                    os.environ["HF_TOKEN"] = token
                                break
                    break  # Successfully read file
                except UnicodeDecodeError:
                    continue  # Try next encoding
        except Exception:
            pass  # Silently continue if .env reading fails

import click


@click.group(
    invoke_without_command=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def cli(ctx):
    """Bruno - Neural behavior engineering framework.

    Named after Giordano Bruno (1548-1600) who revealed infinite
    cosmic worlds against imposed constraints.

    \b
    Commands:
      abliterate    Run abliteration on a model (default)
      chat          Interactive chat with your model
      show-config   Display effective configuration and exit

    \b
    Examples:
      bruno --model Qwen/Qwen2.5-7B-Instruct --n-trials 200
      bruno abliterate --model Qwen/Qwen2.5-7B
      bruno chat --model rawcell/bruno --4bit
      bruno show-config --model Qwen/Qwen2.5-7B
    """
    # If no subcommand provided, run abliterate as default
    if ctx.invoked_subcommand is None:
        ctx.invoke(abliterate)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.pass_context
def abliterate(ctx):
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


@cli.command(
    "show-config",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def show_config(ctx):
    """Display effective configuration and exit.

    Shows all configuration values that would be used for a run,
    combining CLI arguments, environment variables, and config.toml.

    Useful for verifying configuration before starting a long run.

    \b
    Example:
      bruno show-config --model Qwen/Qwen2.5-7B-Instruct
      bruno show-config  # Uses placeholder model
    """
    from pydantic import ValidationError

    from .config import Settings
    from .config_verify import (
        log_config_status,
        print_config_summary,
        verify_config_was_parsed,
    )
    from .utils import print

    # Inject a placeholder model into argv if not already there
    # Settings requires a model, but we just want to show config
    if "--model" not in sys.argv and "-m" not in sys.argv:
        sys.argv.extend(["--model", "placeholder-model"])

    try:
        settings = Settings()
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")
        for err in error.errors():
            print(f"[bold]{err['loc'][0]}[/]: [yellow]{err['msg']}[/]")
        sys.exit(1)

    # Show config status
    config_found = log_config_status()

    # Check for silent failures
    if config_found:
        warnings = verify_config_was_parsed(settings)
        for warning in warnings:
            print(f"[yellow]Warning: {warning}[/yellow]")

    # Print full config summary
    print_config_summary(settings)

    print("[dim]Run 'bruno --help' for usage information.[/dim]")


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
