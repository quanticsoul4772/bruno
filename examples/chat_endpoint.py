#!/usr/bin/env python3
"""Chat with Moonlight via HuggingFace Inference Endpoint.

Simple terminal chat interface for testing the abliterated Moonlight model
deployed on HuggingFace Inference Endpoints.
"""

import os
import sys

import requests
from rich.console import Console

console = Console()


def chat_with_endpoint(
    endpoint_url: str,
    token: str,
    model_name: str = "rawcell/Moonlight-16B-A3B-Instruct-bruno",
):
    """Interactive chat with HuggingFace Inference Endpoint.

    Args:
        endpoint_url: Full endpoint URL (e.g., https://xxx.endpoints.huggingface.cloud)
        token: HuggingFace token for authentication
        model_name: Model name to use in API calls
    """
    # Ensure endpoint URL has /v1/chat/completions
    if not endpoint_url.endswith("/v1/chat/completions"):
        if endpoint_url.endswith("/"):
            endpoint_url = endpoint_url.rstrip("/")
        endpoint_url = f"{endpoint_url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    console.print()
    console.print("[cyan]Moonlight Abliterated Chat[/]")
    console.print(f"[dim]Endpoint: {endpoint_url}[/]")
    console.print(f"[dim]Model: {model_name}[/]")
    console.print()
    console.print("[bold]Commands:[/]")
    console.print("  [bold]/exit[/]  - Exit chat")
    console.print("  [bold]/clear[/] - Clear conversation")
    console.print("  [bold]Ctrl+C[/] - Exit chat")
    console.print()

    # Conversation history
    messages = []

    while True:
        try:
            # Get user input
            console.print()
            console.print("[bold green]You:[/] ", end="")
            user_input = input()

            if not user_input.strip():
                continue

            # Handle commands
            command = user_input.strip().lower()

            if command in ["/exit", "/quit", "/q"]:
                console.print("[dim]Goodbye![/]")
                break

            if command == "/clear":
                messages = []
                console.print("[dim]History cleared.[/]")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Call endpoint
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": False,
            }

            console.print("[bold cyan]Moonlight:[/] ", end="")

            response = requests.post(
                endpoint_url, json=payload, headers=headers, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]

                # Print response
                console.print(assistant_message)

                # Add to history
                messages.append({"role": "assistant", "content": assistant_message})
            else:
                console.print(f"[red]Error {response.status_code}:[/]")
                console.print(f"[red]{response.text}[/]")

        except KeyboardInterrupt:
            console.print()
            console.print("[dim]Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


if __name__ == "__main__":
    # Get endpoint URL from args or env
    endpoint_url = os.environ.get("MOONLIGHT_ENDPOINT_URL")
    if len(sys.argv) > 1:
        endpoint_url = sys.argv[1]

    if not endpoint_url:
        console.print("[red]Error: Endpoint URL required[/]")
        console.print()
        console.print("[yellow]Usage:[/]")
        console.print("  python examples/chat_endpoint.py <endpoint_url>")
        console.print("  OR set MOONLIGHT_ENDPOINT_URL environment variable")
        console.print()
        console.print("[yellow]Example:[/]")
        console.print(
            "  python examples/chat_endpoint.py https://xxx.us-east-1.aws.endpoints.huggingface.cloud"
        )
        sys.exit(1)

    # Get token
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try reading from .env (handle BOM)
        env_file = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_file):
            for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    with open(env_file, encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("HF_TOKEN="):
                                token = (
                                    line.split("=", 1)[1].strip().strip('"').strip("'")
                                )
                                break
                    if token:
                        break
                except UnicodeDecodeError:
                    continue

    if not token:
        console.print("[red]Error: HF_TOKEN not found[/]")
        console.print(
            "[yellow]Set HF_TOKEN environment variable or add to .env file[/]"
        )
        sys.exit(1)

    chat_with_endpoint(endpoint_url, token)
