"""Progress tracking utilities for long operations."""

from contextlib import contextmanager
from typing import Optional

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .utils import print as rich_print


@contextmanager
def operation_progress(
    description: str, total: Optional[int] = None, unit: str = "items"
):
    """
    Context manager for operation progress tracking.

    Args:
        description: Operation description
        total: Total items (None for indeterminate spinner)
        unit: Unit name for progress display

    Yields:
        Progress updater function
    """
    if total is not None:
        # Determinate progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

        progress.start()
        task_id = progress.add_task(description, total=total)

        def update(advance: int = 1):
            progress.update(task_id, advance=advance)

        try:
            yield update
        finally:
            progress.stop()
            rich_print(f"[green]{description} complete[/]")
    else:
        # Indeterminate spinner
        rich_print(f"[cyan]{description}...[/]")

        def update(advance: int = 1):
            pass  # No-op for spinner

        try:
            yield update
        finally:
            rich_print(f"[green]{description} complete[/]")


@contextmanager
def batch_progress(description: str, total_batches: int, batch_size: int):
    """
    Progress tracking for batched operations.

    Args:
        description: Operation description
        total_batches: Total number of batches
        batch_size: Items per batch

    Yields:
        Progress updater function
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total} batches)"),
        TimeRemainingColumn(),
    )

    progress.start()
    task_id = progress.add_task(description, total=total_batches)

    def update(batch_idx: int):
        progress.update(
            task_id,
            completed=batch_idx + 1,
            description=f"{description} [batch {batch_idx + 1}/{total_batches}]",
        )

    try:
        yield update
    finally:
        progress.stop()
        total_items = total_batches * batch_size
        rich_print(f"[green]✓[/] {description} complete ({total_items} items)")


@contextmanager
def layer_progress(description: str, total_layers: int):
    """
    Progress tracking for layer-wise operations.

    Args:
        description: Operation description
        total_layers: Total number of layers

    Yields:
        Progress updater function
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total} layers)"),
        TimeRemainingColumn(),
    )

    progress.start()
    task_id = progress.add_task(description, total=total_layers)

    def update(layer_idx: int):
        progress.update(
            task_id,
            completed=layer_idx + 1,
            description=f"{description} [layer {layer_idx + 1}/{total_layers}]",
        )

    try:
        yield update
    finally:
        progress.stop()
        rich_print(f"[green]✓[/] {description} complete ({total_layers} layers)")
