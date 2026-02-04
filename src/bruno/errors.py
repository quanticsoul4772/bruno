"""Error message formatting utilities."""

from typing import Optional

from .utils import print as rich_print


def format_error_message(
    operation: str,
    reason: str,
    impact: str,
    solutions: list[str],
    context: Optional[dict] = None,
) -> str:
    """
    Format comprehensive error message with context.

    Args:
        operation: What operation failed
        reason: Why it failed (root cause)
        impact: What won't work without this
        solutions: List of actionable solutions
        context: Optional additional context (e.g., metrics, values)

    Returns:
        Formatted error message
    """
    lines = [
        f"[red]{operation} failed[/]",
        f"[yellow]Reason:[/] {reason}",
        f"[yellow]Impact:[/] {impact}",
        "[yellow]Solutions:[/]",
    ]

    for i, solution in enumerate(solutions, 1):
        lines.append(f"  {i}. {solution}")

    if context:
        lines.append("\n[dim]Additional Context:[/]")
        for key, value in context.items():
            lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def print_error(
    operation: str,
    reason: str,
    impact: str,
    solutions: list[str],
    context: Optional[dict] = None,
):
    """Print formatted error message."""
    message = format_error_message(operation, reason, impact, solutions, context)
    rich_print(message)


def format_warning_message(operation: str, issue: str, recommendation: str) -> str:
    """
    Format warning message.

    Args:
        operation: What operation triggered warning
        issue: What the issue is
        recommendation: What to do about it

    Returns:
        Formatted warning message
    """
    return (
        f"[yellow]{operation} warning[/]\n"
        f"[yellow]Issue:[/] {issue}\n"
        f"[yellow]Recommendation:[/] {recommendation}"
    )


def print_warning(operation: str, issue: str, recommendation: str):
    """Print formatted warning message."""
    message = format_warning_message(operation, issue, recommendation)
    rich_print(message)
