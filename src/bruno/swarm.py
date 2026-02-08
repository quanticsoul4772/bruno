# SPDX-License-Identifier: AGPL-3.0-or-later
# Bruno AI Developer Swarm CLI
#
# Production CLI for running multi-agent development swarms
# powered by abliterated Bruno models via CrewAI + Ollama.
#
# CrewAI is an optional dependency: pip install bruno-ai[swarm]

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .logging import get_logger

logger = get_logger(__name__)

console = Console()

# Default Ollama URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Agent configurations: name -> (ollama_model, role, goal, backstory)
AGENT_CONFIGS = {
    "orchestrator": {
        "model": "orchestrator",
        "role": "Senior Software Architect",
        "goal": "Plan development tasks, design system architecture, and coordinate the team",
        "backstory": (
            "Senior architect with 20 years of experience. Thinks step by step, "
            "breaks complex problems into clear tasks, and delegates to specialists. "
            "Reviews all work for quality and architectural consistency."
        ),
        "allow_delegation": True,
    },
    "frontend": {
        "model": "frontend",
        "role": "Frontend Developer",
        "goal": "Build responsive, user-friendly React components with TypeScript",
        "backstory": (
            "Expert in React, TypeScript, Tailwind CSS. Writes clean, concise "
            "code without over-engineering. Focuses on accessibility and UX."
        ),
        "allow_delegation": False,
    },
    "backend": {
        "model": "backend",
        "role": "Backend Developer",
        "goal": "Create scalable FastAPI endpoints and database schemas",
        "backstory": (
            "Expert in FastAPI, PostgreSQL, async patterns. Focuses on clean "
            "architecture without premature optimization. Writes clear API contracts."
        ),
        "allow_delegation": False,
    },
    "test": {
        "model": "test",
        "role": "QA Engineer",
        "goal": "Write comprehensive test suites with high coverage",
        "backstory": (
            "Expert in pytest, coverage analysis, edge cases. Proactively writes "
            "tests for all code paths including error handling and boundary conditions."
        ),
        "allow_delegation": False,
    },
    "security": {
        "model": "security",
        "role": "Security Engineer",
        "goal": "Identify vulnerabilities and enforce secure coding practices",
        "backstory": (
            "Expert in OWASP Top 10, penetration testing, secure code review. "
            "Paranoid about security -- catches issues others miss. Reviews all "
            "code for injection, auth bypass, and data exposure risks."
        ),
        "allow_delegation": False,
    },
    "docs": {
        "model": "docs",
        "role": "Technical Writer",
        "goal": "Write clear API docs, README files, and developer guides",
        "backstory": (
            "Expert in technical documentation, API references, and developer "
            "onboarding. Writes concise docs without unnecessary jargon. "
            "Focuses on examples and practical usage."
        ),
        "allow_delegation": False,
    },
    "devops": {
        "model": "devops",
        "role": "DevOps Engineer",
        "goal": "Create Docker configs, CI/CD pipelines, and deployment scripts",
        "backstory": (
            "Expert in Docker, GitHub Actions, infrastructure as code. "
            "Writes practical deployment configurations without overengineering. "
            "Focuses on reproducibility and security."
        ),
        "allow_delegation": False,
    },
}

# Specialist agent names (all except orchestrator)
SPECIALISTS = ["frontend", "backend", "test", "security", "docs", "devops"]


def _check_crewai():
    """Check that CrewAI is installed, exit with install instructions if not."""
    try:
        import crewai  # noqa: F401

        return True
    except ImportError:
        console.print("[red]Error: CrewAI is not installed.[/]")
        console.print()
        console.print("Install it with:")
        console.print("  [cyan]pip install bruno-ai[swarm][/]")
        console.print("  [dim]or[/]")
        console.print("  [cyan]uv sync --extra swarm[/]")
        sys.exit(1)


def _parse_agents(agents_str: str | None) -> list[str] | None:
    """Parse comma-separated agent names, validate, and return list or None."""
    if not agents_str:
        return None

    agent_names = [a.strip() for a in agents_str.split(",")]
    invalid = [a for a in agent_names if a not in AGENT_CONFIGS]
    if invalid:
        console.print(f"[red]Unknown agents: {invalid}[/]")
        console.print(f"Available: {', '.join(AGENT_CONFIGS.keys())}")
        sys.exit(1)

    return agent_names


def create_llm(model_name: str, base_url: str):
    """Create a CrewAI LLM instance for an Ollama model."""
    from crewai import LLM

    return LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        timeout=1200,
        max_retries=3,
    )


def create_agent(name: str, base_url: str):
    """Create a CrewAI agent from config."""
    from crewai import Agent

    config = AGENT_CONFIGS[name]
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=create_llm(config["model"], base_url),
        verbose=True,
        allow_delegation=config["allow_delegation"],
        max_iter=10,
        max_retry_limit=3,
    )


def create_hierarchical_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
):
    """Create a hierarchical crew with 14B orchestrator as manager.

    The orchestrator plans and delegates to specialist agents.
    """
    from crewai import Crew, Process, Task

    if agent_names is None:
        agent_names = SPECIALISTS

    # Create orchestrator as manager (must NOT be in agents list)
    manager = create_agent("orchestrator", base_url)

    # Create specialist agents
    agents = [create_agent(name, base_url) for name in agent_names]

    # Single high-level task -- orchestrator delegates to specialists
    task = Task(
        description=(
            f"{task_description}\n\n"
            "Break this into subtasks and delegate to the appropriate specialists. "
            "Each specialist should return ONLY code, no explanations. "
            "Review all outputs for quality and consistency before finalizing."
        ),
        expected_output="Complete implementation with all components integrated",
    )

    return Crew(
        agents=agents,
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def create_flat_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
):
    """Create a flat sequential crew without orchestrator.

    Each specialist handles their portion of the task sequentially.
    """
    from crewai import Crew, Process, Task

    if agent_names is None:
        agent_names = SPECIALISTS

    agents = {}
    for name in agent_names:
        agents[name] = create_agent(name, base_url)

    # Role-specific task templates
    task_templates = {
        "backend": (
            "Design and implement the backend for: {task}\n"
            "Include API endpoints, schemas, and database models. "
            "Return ONLY the code, no explanations."
        ),
        "frontend": (
            "Build the frontend components for: {task}\n"
            "Use React with TypeScript and Tailwind CSS. "
            "Return ONLY the code, no explanations."
        ),
        "test": (
            "Write comprehensive tests for: {task}\n"
            "Use pytest with fixtures. Cover happy paths, edge cases, and error handling. "
            "Return ONLY the code, no explanations."
        ),
        "security": (
            "Perform a security review of the implementation for: {task}\n"
            "Check for OWASP Top 10 vulnerabilities, auth issues, injection risks. "
            "Return findings and fixed code snippets."
        ),
        "docs": (
            "Write documentation for: {task}\n"
            "Include API reference, setup guide, and usage examples. "
            "Return ONLY the documentation in Markdown."
        ),
        "devops": (
            "Create deployment configuration for: {task}\n"
            "Include Dockerfile, docker-compose.yml, and CI/CD pipeline. "
            "Return ONLY the configuration files."
        ),
    }

    expected_outputs = {
        "backend": "Complete backend code with API endpoints and schemas",
        "frontend": "Complete React components with TypeScript types",
        "test": "Complete pytest test suite with fixtures and assertions",
        "security": "Security audit report with vulnerability fixes",
        "docs": "Complete documentation in Markdown format",
        "devops": "Dockerfile, docker-compose.yml, and CI/CD config",
    }

    tasks = []
    for name in agent_names:
        if name in task_templates:
            tasks.append(
                Task(
                    description=task_templates[name].format(task=task_description),
                    agent=agents[name],
                    expected_output=expected_outputs[name],
                )
            )

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


# CLI Commands
@click.group()
def cli():
    """Bruno AI Developer Swarm CLI.

    Multi-agent development team powered by abliterated Bruno models.
    Uses CrewAI for orchestration and Ollama for local inference.

    Install CrewAI: pip install bruno-ai[swarm]
    """
    pass


@cli.command("run")
@click.option("--task", "-t", required=True, help="Development task to execute")
@click.option("--flat", is_flag=True, help="Use flat sequential mode (no orchestrator)")
@click.option(
    "--agents",
    "-a",
    default=None,
    help="Comma-separated agent names (default: all specialists)",
)
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Save result to file",
)
def run_task(
    task: str, flat: bool, agents: str | None, ollama_url: str, output: str | None
):
    """Execute a development task with the agent swarm."""
    _check_crewai()

    # Disable CrewAI tracing in non-interactive mode
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    agent_names = _parse_agents(agents)
    mode = "flat" if flat else "hierarchical"
    agents_str = ", ".join(agent_names) if agent_names else "all specialists"

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Bruno AI Developer Swarm[/]\n\n"
            f"Mode: [cyan]{mode}[/]\n"
            f"Agents: [cyan]{agents_str}[/]\n"
            f"Ollama: [cyan]{ollama_url}[/]\n"
            f"Task: [cyan]{task}[/]",
            title="Swarm",
            border_style="cyan",
        )
    )

    try:
        if flat:
            crew = create_flat_crew(task, ollama_url, agent_names)
        else:
            crew = create_hierarchical_crew(task, ollama_url, agent_names)

        result = crew.kickoff()

        console.print()
        console.print(
            Panel(
                str(result),
                title="[bold green]Swarm Result[/]",
                border_style="green",
            )
        )

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(str(result), encoding="utf-8")
            console.print(f"\nResult saved to [cyan]{output_path}[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        logger.error("Swarm execution failed", exc_info=True)
        console.print(f"\n[red]Swarm failed: {type(e).__name__}: {e}[/]")
        sys.exit(1)


@cli.command("agents")
def list_agents():
    """List all available swarm agents and their roles."""
    table = Table(title="Bruno Swarm Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="bold")
    table.add_column("Ollama Model", style="dim")
    table.add_column("Goal")

    for name, config in AGENT_CONFIGS.items():
        table.add_row(
            name,
            config["role"],
            config["model"],
            config["goal"],
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Specialists (flat mode):[/]", ", ".join(SPECIALISTS))
    console.print(
        "[dim]Hierarchical mode uses orchestrator as manager + specialists[/]"
    )


@cli.command("status")
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
def check_status(ollama_url: str):
    """Check Ollama connectivity and loaded models."""
    import urllib.error
    import urllib.request

    console.print()
    console.print(f"Checking Ollama at [cyan]{ollama_url}[/]...")
    console.print()

    # Check Ollama connectivity
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json

            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        console.print(f"[red]Cannot connect to Ollama: {e.reason}[/]")
        console.print()
        console.print("Make sure Ollama is running:")
        console.print("  [cyan]ollama serve[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/]")
        sys.exit(1)

    console.print("[green]Ollama is running[/]")
    console.print()

    # List available models and check which swarm agents are present
    models = data.get("models", [])
    model_names = {m.get("name", "").split(":")[0] for m in models}

    table = Table(title="Swarm Agent Models")
    table.add_column("Agent", style="cyan")
    table.add_column("Ollama Model")
    table.add_column("Status")
    table.add_column("Size", justify="right")

    for name, config in AGENT_CONFIGS.items():
        ollama_model = config["model"]
        if ollama_model in model_names:
            # Find the model entry for size info
            size_str = ""
            for m in models:
                if m.get("name", "").split(":")[0] == ollama_model:
                    size_bytes = m.get("size", 0)
                    if size_bytes > 0:
                        size_str = f"{size_bytes / (1024**3):.1f} GB"
                    break
            table.add_row(name, ollama_model, "[green]available[/]", size_str)
        else:
            table.add_row(name, ollama_model, "[red]missing[/]", "")

    console.print(table)

    # Summary
    available = sum(1 for c in AGENT_CONFIGS.values() if c["model"] in model_names)
    total = len(AGENT_CONFIGS)
    console.print()
    if available == total:
        console.print(f"[green]All {total} agent models available[/]")
    else:
        console.print(f"[yellow]{available}/{total} agent models available[/]")
        missing = [n for n, c in AGENT_CONFIGS.items() if c["model"] not in model_names]
        console.print(f"[dim]Missing: {', '.join(missing)}[/]")

    # Show loaded models (currently in memory)
    try:
        req = urllib.request.Request(f"{ollama_url}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json

            ps_data = json.loads(resp.read().decode())
            running = ps_data.get("models", [])
            if running:
                console.print()
                console.print("[bold]Currently loaded in memory:[/]")
                for m in running:
                    name = m.get("name", "unknown")
                    size_bytes = m.get("size", 0)
                    size_str = f"{size_bytes / (1024**3):.1f} GB" if size_bytes else ""
                    console.print(f"  [green]{name}[/] {size_str}")
    except Exception:
        pass  # /api/ps may not be available in older Ollama versions


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
