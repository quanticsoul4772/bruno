# Bruno Chat Command Integration - Technical Explanation

**Goal:** Add `bruno chat` to the existing `bruno` CLI
**Result:** One tool for both abliteration and interactive chat

---

## Current Bruno CLI Architecture

### How `bruno` Works Now

**Entry Point:** `pyproject.toml`
```toml
[project.scripts]
bruno = "bruno.main:main"
```

**When you run `bruno`:**
```
bruno --model Qwen/Qwen2.5-7B --n-trials 200
```

**What happens:**
1. Python calls `bruno.main:main()` function
2. `main()` calls `run()` function
3. `run()` uses Pydantic Settings to parse CLI args
4. Settings object created with all flags
5. Abliteration pipeline executes

**Current structure:**
```
bruno (command)
  └─> main()
       └─> run()
            └─> Abliteration pipeline
```

---

## Proposed Architecture: Add Subcommands

### Option A: Use Click Framework (Recommended)

**Transform to subcommand structure:**

```
bruno (group)
  ├─> abliterate (default subcommand)
  └─> chat (new subcommand)
```

**Implementation:**

**File:** `src/bruno/cli.py` (new file)
```python
import click
from .main import run as run_abliterate
from .chat import run_chat

@click.group()
def cli():
    """Bruno - Neural behavior engineering framework.

    Named after Giordano Bruno (1548-1600).
    """
    pass

@cli.command()
@click.pass_context
def abliterate(ctx):
    """Run abliteration on a model (default command)."""
    # Call existing run() function
    run_abliterate()

@cli.command()
@click.option('--model', default='rawcell/bruno', help='Model to chat with')
@click.option('--4bit', is_flag=True, help='Use 4-bit quantization')
@click.option('--temperature', default=0.7, help='Sampling temperature')
@click.option('--max-tokens', default=2048, help='Max tokens per response')
def chat(model, 4bit, temperature, max_tokens):
    """Interactive chat with your abliterated model.

    Example:
        bruno chat
        bruno chat --model rawcell/qwen-7b-abliterated --4bit
    """
    run_chat(
        model_name=model,
        use_4bit=4bit,
        temperature=temperature,
        max_new_tokens=max_tokens
    )

# Make abliterate the default command
cli.add_command(abliterate, name='abliterate')
```

**Update:** `pyproject.toml`
```toml
[project.scripts]
bruno = "bruno.cli:cli"  # Changed from bruno.main:main
bruno-vast = "bruno.vast:main"
```

**Usage:**
```bash
# Abliteration (default subcommand)
bruno abliterate --model Qwen/Qwen2.5-7B --n-trials 200

# Or shorter (abliterate is default)
bruno --model Qwen/Qwen2.5-7B --n-trials 200

# Chat
bruno chat
bruno chat --model rawcell/bruno --4bit
bruno chat --help
```

---

### Option B: Use Typer Framework (Modern Alternative)

**Similar to Click but with type hints:**

```python
import typer
from typing_extensions import Annotated

app = typer.Typer(
    help="Bruno - Neural behavior engineering framework"
)

@app.command()
def abliterate(
    model: Annotated[str, typer.Option(help="Model to abliterate")],
    n_trials: Annotated[int, typer.Option(help="Number of trials")] = 200,
):
    """Run abliteration on a model."""
    from .main import run
    run()

@app.command()
def chat(
    model: Annotated[str, typer.Option(help="Model to chat with")] = "rawcell/bruno",
    use_4bit: Annotated[bool, typer.Option("--4bit", help="Use 4-bit quant")] = False,
):
    """Interactive chat with your model."""
    from .chat import run_chat
    run_chat(model_name=model, use_4bit=use_4bit)
```

**Pros:** Modern, uses type hints, auto-generates help
**Cons:** Another dependency

---

### Option C: Simple Flag-Based (Minimal Changes)

**Add `--chat` flag to existing CLI:**

**File:** `src/bruno/config.py`
```python
class Settings(BaseSettings):
    # Existing settings...

    # Chat mode
    chat: bool = Field(
        default=False,
        description="Enter interactive chat mode instead of abliteration"
    )

    chat_model: str = Field(
        default="rawcell/bruno",
        description="Model to use for chat mode"
    )
```

**File:** `src/bruno/main.py`
```python
def main():
    install()  # Rich traceback

    try:
        settings = Settings()  # Parse CLI args

        if settings.chat:
            # Launch chat mode
            from .chat import run_chat
            run_chat(
                model_name=settings.chat_model,
                use_4bit=settings.device_map == "auto"
            )
        else:
            # Run normal abliteration
            run()
    except ...
```

**Usage:**
```bash
# Abliteration (default)
bruno --model Qwen/Qwen2.5-7B

# Chat mode
bruno --chat
bruno --chat --chat-model rawcell/bruno
```

**Pros:** Minimal code changes, uses existing Settings
**Cons:** Less clean CLI UX, flags get messy

---

## The Chat Module Implementation

**File:** `src/bruno/chat.py` (new file)

```python
"""Interactive chat interface for abliterated models."""

import sys
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

from .logging import get_logger
from .utils import empty_cache, print

logger = get_logger(__name__)


class BrunoChat:
    """Interactive chat session with an abliterated model."""

    def __init__(
        self,
        model_name: str,
        use_4bit: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ):
        """Initialize chat session.

        Args:
            model_name: HuggingFace model ID or local path
            use_4bit: Use 4-bit quantization (for 8GB VRAM)
            temperature: Sampling temperature (0.0-1.0)
            max_new_tokens: Max tokens per response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.history: list[dict] = []

        print(f"Loading [bold]{model_name}[/]...")

        # Load model
        quantization_config = None
        if use_4bit:
            print("  * Using 4-bit quantization for RTX 4070")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Setup streaming output
        self.streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        print("[green]✓[/] Bruno ready!")
        print()
        print("[dim]Commands:[/]")
        print("  [bold]/clear[/]  - Clear conversation history")
        print("  [bold]/exit[/]   - Exit chat")
        print("  [bold]Ctrl+C[/]  - Exit chat")
        print()

    def chat(self, user_message: str) -> str:
        """Send message and get streaming response.

        Args:
            user_message: User's message

        Returns:
            Model's response text
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        # Generate with streaming
        print("[bold cyan]Bruno:[/] ", end="")

        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            streamer=self.streamer,  # Enable streaming output
        )

        # Extract response text
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1] :],
            skip_special_tokens=True,
        )

        # Add to history
        self.history.append({"role": "assistant", "content": response})

        return response

    def run_interactive(self):
        """Run interactive chat loop."""
        while True:
            try:
                # Get user input
                print()
                user_input = input("[bold green]You:[/] ")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.strip().lower() in ["/exit", "/quit", "/q"]:
                    print("[dim]Goodbye![/]")
                    break

                if user_input.strip().lower() == "/clear":
                    self.history = []
                    empty_cache()
                    print("[dim]History cleared.[/]")
                    continue

                # Chat
                self.chat(user_input)

            except KeyboardInterrupt:
                print()
                print("[dim]Goodbye![/]")
                break

            except Exception as e:
                logger.error(f"Chat error: {e}")
                print(f"\n[red]Error: {e}[/]")
                print("[yellow]Continuing...[/]")


def run_chat(
    model_name: str = "rawcell/bruno",
    use_4bit: bool = False,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
):
    """Entry point for chat mode.

    Args:
        model_name: HuggingFace model ID or local path
        use_4bit: Use 4-bit quantization
        temperature: Sampling temperature
        max_new_tokens: Max tokens per response
    """
    # Print banner
    print()
    print("[cyan]█▀▄░█▀▄░█░█░█▄░█░█▀█[/]  Chat Mode")
    print("[cyan]█▀▄░█▀▄░█░█░█░▀█░█░█[/]")
    print("[cyan]▀▀░░▀░▀░▀▀▀░▀░░▀░▀▀▀[/]  Interactive chat with your model")
    print()

    # Create and run chat session
    chat_session = BrunoChat(
        model_name=model_name,
        use_4bit=use_4bit,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    chat_session.run_interactive()
```

---

## How the Integration Works (Step by Step)

### Step 1: User Runs Command

```bash
bruno chat --model rawcell/bruno --4bit
```

### Step 2: Python Entry Point

```
Python interpreter
  └─> Reads pyproject.toml: [project.scripts]
       └─> bruno = "bruno.cli:cli"
            └─> Calls cli() function in bruno/cli.py
```

### Step 3: Click Parses Command

```python
# cli.py
@click.group()
def cli():
    pass

@cli.command()
def chat(model, 4bit, ...):
    run_chat(model, 4bit)
```

**Click framework:**
1. Parses `chat` as subcommand
2. Parses `--model rawcell/bruno` as option
3. Parses `--4bit` as flag
4. Calls `chat()` function with parsed args

### Step 4: Chat Module Executes

```python
# chat.py
def run_chat(model_name, use_4bit):
    # 1. Load model
    model = AutoModelForCausalLM.from_pretrained(...)

    # 2. Create chat session
    session = BrunoChat(model, ...)

    # 3. Start interactive loop
    session.run_interactive()
        while True:
            user_input = input("You: ")
            response = session.chat(user_input)
            print(f"Bruno: {response}")
```

### Step 5: Interactive Loop

```
User types: "Write Python code"
  └─> chat() method:
       ├─> Add to history: [{"role": "user", "content": "..."}]
       ├─> Apply chat template
       ├─> model.generate() with streaming
       ├─> Response streams to terminal
       └─> Add response to history

User types: "Make it faster"
  └─> Uses conversation history for context
       └─> Generates improved version
```

---

## Code Flow Diagram

```
Terminal
   │
   ├─ bruno abliterate --model X --n-trials 200
   │    └─> bruno.cli:cli() → abliterate command → main.run()
   │         └─> Abliteration pipeline (current behavior)
   │
   └─ bruno chat --model rawcell/bruno --4bit
        └─> bruno.cli:cli() → chat command → chat.run_chat()
             └─> BrunoChat class
                  ├─> Load model with quantization
                  ├─> Initialize streamer
                  └─> Interactive loop:
                       ├─> input("You: ")
                       ├─> Apply chat template
                       ├─> model.generate(streamer=TextStreamer)
                       ├─> Response streams to terminal
                       └─> Repeat
```

---

## File Changes Required

### 1. Create `src/bruno/cli.py` (NEW FILE)

**Purpose:** Command routing for subcommands
**Size:** ~50 lines
**Content:**
- Click group definition
- `abliterate` command (calls existing run())
- `chat` command (calls new run_chat())

### 2. Create `src/bruno/chat.py` (NEW FILE)

**Purpose:** Interactive chat functionality
**Size:** ~200 lines
**Content:**
- `BrunoChat` class
- Model loading with quantization
- Interactive loop with streaming
- History management
- Command handling (/clear, /exit)

### 3. Update `pyproject.toml`

**Change:**
```toml
# OLD
bruno = "bruno.main:main"

# NEW
bruno = "bruno.cli:cli"
```

### 4. Add Click Dependency

**Update:** `pyproject.toml` dependencies
```toml
dependencies = [
    # ... existing ...
    "click>=8.1.8",  # Already there for vast.py
]
```

**No new dependencies needed!** Click is already required for bruno-vast.

---

## User Experience Comparison

### Before (Two Separate Tools)

```bash
# Abliterate
bruno --model Qwen/Qwen2.5-7B --n-trials 200

# Chat (separate app)
python examples/chat_app.py
# Or use external tool like aichat
```

### After (Integrated)

```bash
# Abliterate (unchanged)
bruno --model Qwen/Qwen2.5-7B --n-trials 200

# Chat (integrated!)
bruno chat
bruno chat --model rawcell/bruno --4bit

# Both from same CLI tool
```

---

## Advanced Features (Possible Extensions)

### Feature 1: A/B Testing

```bash
# Chat with original
bruno chat --model Qwen/Qwen2.5-7B-Instruct

# Chat with abliterated
bruno chat --model rawcell/bruno

# Compare side-by-side
bruno compare --model1 Qwen/Qwen2.5-7B --model2 rawcell/bruno
```

### Feature 2: Evaluation Mode

```bash
# Test abliterated model on refusal prompts
bruno chat --model rawcell/bruno --eval-mode
# Loads bad_prompts, shows refusal rate
```

### Feature 3: Save Conversations

```bash
# Auto-save conversations
bruno chat --save-to ./conversations/

# Load previous conversation
bruno chat --load ./conversations/session_2026-01-31.json
```

### Feature 4: Streaming to File

```bash
# Generate documentation and save
bruno chat --output code.py
You: Write a complete REST API
Bruno: [streams code to code.py]
```

---

## Implementation Complexity

| Task | Complexity | Time | Dependencies |
|------|-----------|------|--------------|
| Create cli.py | Low | 30 min | click (already installed) |
| Create chat.py | Medium | 2 hours | transformers (already installed) |
| Update pyproject.toml | Low | 5 min | None |
| Add tests | Medium | 1 hour | pytest (already installed) |
| **Total** | **Medium** | **~4 hours** | **None (all deps exist)** |

---

## Benefits of Integration

**1. Single Tool**
- One CLI for everything
- Consistent interface
- Easier to remember

**2. Shared Code**
- Reuse model loading logic
- Same configuration system
- Consistent error handling

**3. Workflow Efficiency**
```bash
# Abliterate, then immediately test
bruno --model X --n-trials 200 --auto-select
bruno chat --model ./models/X-bruno

# All in one tool
```

**4. Testing Integration**
```bash
# Abliterate model
bruno --model X

# Test it immediately
bruno chat --model ./models/X-bruno --eval-mode

# No context switching
```

---

## Alternative: Keep Separate (Simpler)

**If you don't want CLI changes:**

**Create standalone:** `scripts/bruno-chat`

```bash
#!/usr/bin/env python3
# Standalone chat script (no bruno CLI changes)

from bruno.chat import run_chat
run_chat()
```

**Make executable:**
```bash
chmod +x scripts/bruno-chat
sudo ln -s $(pwd)/scripts/bruno-chat /usr/local/bin/bruno-chat
```

**Usage:**
```bash
bruno-chat  # Separate command
bruno --model X  # Original bruno unchanged
```

**Pros:** No changes to bruno CLI, zero risk
**Cons:** Two separate commands

---

## My Recommendation

### Use Option A (Click Subcommands)

**Why:**
1. Professional CLI UX (`bruno chat` feels natural)
2. Click already installed (for bruno-vast)
3. Easy to extend later (bruno compare, bruno eval, etc.)
4. Standard pattern (git, docker, npm all use this)
5. Clear help system (`bruno --help`, `bruno chat --help`)

**Implementation steps:**
1. Create `src/bruno/cli.py` (routing)
2. Create `src/bruno/chat.py` (chat logic)
3. Update `pyproject.toml` (one line change)
4. Test both commands work

**Time:** ~4 hours total

**Result:**
```bash
$ bruno chat
Loading rawcell/bruno...
✓ Bruno ready!

Commands:
  /clear  - Clear conversation history
  /exit   - Exit chat
  Ctrl+C  - Exit chat

You: Write Python code for quicksort

Bruno: Here's an efficient quicksort implementation:
[streams response with syntax highlighting]

You: Make it handle duplicates better

Bruno: I'll modify it to use 3-way partitioning:
[continues conversation with context]
```

---

## Want Me to Implement It?

I can implement Option A (Click subcommands) with:
- `bruno chat` command
- Streaming responses
- Conversation history
- 4-bit quantization support
- Clean terminal UX

**Ready to proceed?**
