#!/usr/bin/env python3
"""Patch heretic to add --auto-select flag"""

from pathlib import Path

HERETIC_PATH = "/usr/local/lib/python3.11/dist-packages/heretic"

# Patch config.py
config_path = Path(HERETIC_PATH) / "config.py"
content = config_path.read_text()

auto_select_code = """

    auto_select: bool = Field(
        default=False,
        description="Automatically select the best trial (lowest refusals) and save without interactive prompts.",
    )

    auto_select_path: str | None = Field(
        default=None,
        description="Path to save the model when using --auto-select.",
    )
"""

if "auto_select" not in content:
    # Find the end of evaluate_model field and insert after it
    marker = '        description="If this model ID or path is set, then instead of abliterating the main model, evaluate this model relative to the main model.",\n    )'
    content = content.replace(marker, marker + auto_select_code)
    config_path.write_text(content)
    print("config.py patched!")
else:
    print("auto_select already in config.py")

# Patch main.py - add auto-select logic before interactive mode
main_path = Path(HERETIC_PATH) / "main.py"
main_content = main_path.read_text()

auto_select_main_code = """
    # Auto-select mode: automatically pick best trial and save without prompts
    if settings.auto_select:
        if not best_trials:
            print("[red]No Pareto-optimal trials found. Cannot auto-select.[/]")
            return

        # Select the best trial (first one = lowest refusals)
        trial = best_trials[0]
        print(
            f"[cyan]Auto-selecting trial {trial.user_attrs['index']} "
            f"(Refusals: {trial.user_attrs['refusals']}, "
            f"KL divergence: {trial.user_attrs['kl_divergence']:.2f})[/]"
        )

        print()
        print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
        print("* Reloading model...")
        model.reload_model()
        print("* Abliterating...")
        model.abliterate(
            refusal_directions,
            trial.user_attrs["direction_index"],
            trial.user_attrs["parameters"],
        )

        # Determine save path
        if settings.auto_select_path:
            save_directory = settings.auto_select_path
        else:
            # Default to model name with -heretic suffix
            model_name = Path(settings.model).name
            save_directory = f"./{model_name}-heretic"

        print(f"Saving model to [bold]{save_directory}[/]...")
        model.model.save_pretrained(save_directory)
        model.tokenizer.save_pretrained(save_directory)
        print(f"[bold green]Model saved to {save_directory}[/]")
        return

"""

if "auto_select" not in main_content:
    # Find the marker and insert auto-select code before interactive mode
    marker = (
        '    print("[bold green]Optimization finished![/]")\n    print()\n\n    print('
    )
    insert_point = '    print("[bold green]Optimization finished![/]")\n    print()\n'
    main_content = main_content.replace(
        insert_point, insert_point + auto_select_main_code
    )
    main_path.write_text(main_content)
    print("main.py patched!")
else:
    print("auto_select already in main.py")

print("Done! Run: heretic --help | grep auto")
