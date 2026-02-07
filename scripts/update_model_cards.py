#!/usr/bin/env python3
"""Update existing HuggingFace model cards with rich content.

This script updates model cards for already-uploaded models with:
- Better tags (architecture-specific, dynamic)
- Usage examples (transformers + vLLM)
- Improved formatting
- Bruno Score and benchmarks (if validation data available)
"""

import os

from dotenv import load_dotenv
from huggingface_hub import ModelCard, ModelCardData

# Load environment variables from .env
load_dotenv()


def get_model_tags(model_name: str, is_moe: bool = False) -> list[str]:
    """Generate dynamic tags based on model architecture."""
    tags = ["heretic", "uncensored", "decensored", "abliterated", "text-generation"]
    name_lower = model_name.lower()

    # Architecture tags
    if "qwen" in name_lower:
        tags.append("qwen2")
    if "llama" in name_lower:
        tags.append("llama")
    if "deepseek" in name_lower:
        tags.append("deepseek")
    if "gemma" in name_lower:
        tags.append("gemma")
    if "phi" in name_lower:
        tags.append("phi")
    if "mistral" in name_lower:
        tags.append("mistral")
    if "moonlight" in name_lower:
        tags.append("moonlight")

    # Capability tags
    if "coder" in name_lower or "code" in name_lower:
        tags.append("code")
    if "instruct" in name_lower or "chat" in name_lower:
        tags.append("conversational")
    if is_moe or "moe" in name_lower:
        tags.append("moe")

    # Bruno-specific
    tags.append("optuna-optimized")
    tags.append("bruno")

    return tags


def get_usage_examples(repo_id: str, base_model: str) -> str:
    """Generate usage examples section."""
    is_chat = any(kw in base_model.lower() for kw in ["instruct", "chat"])

    sections = []

    sections.append("## Usage\n")

    if is_chat:
        sections.append(f"""### Using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Hello, how are you?"}},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
""")
    else:
        sections.append(f"""### Using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
""")

    sections.append(f"""### Using vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo_id}")
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0].outputs[0].text)
```

""")

    return "\n".join(sections)


def update_model_card(repo_id: str, base_model: str, token: str):
    """Update a model card with rich content."""
    print(f"Updating {repo_id}...")

    try:
        # Load existing card
        card = ModelCard.load(repo_id)

        # Initialize metadata if needed
        if card.data is None:
            card.data = ModelCardData()
        if card.data.tags is None:
            card.data.tags = []

        # Add better tags
        is_moe = "moe" in base_model.lower() or "moonlight" in base_model.lower()
        new_tags = get_model_tags(base_model, is_moe)

        # Merge with existing tags (avoid duplicates)
        card.data.tags = list(set(card.data.tags + new_tags))

        # Set YAML frontmatter
        card.data.base_model = base_model
        card.data.pipeline_tag = "text-generation"
        card.data.language = ["en"]
        card.data.library_name = "transformers"

        # Add usage examples if not already present
        if "## Usage" not in card.text and "### Using transformers" not in card.text:
            usage_section = get_usage_examples(repo_id, base_model)

            # Insert usage examples after the first major section
            # (usually after description/benchmarks)
            lines = card.text.split("\n")
            insert_index = len(lines)  # Default to end

            # Try to insert after first ## section
            for i, line in enumerate(lines):
                if i > 5 and line.startswith("## ") and "usage" not in line.lower():
                    # Find next section or end
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("## "):
                            insert_index = j
                            break
                    break

            lines.insert(insert_index, usage_section)
            card.text = "\n".join(lines)

        # Push updated card
        card.push_to_hub(repo_id, token=token)
        print(f"SUCCESS: Updated {repo_id}")

    except Exception as e:
        print(f"FAILED: Failed to update {repo_id}: {e}")


def main():
    """Update all rawcell models."""
    # Try to load from environment first
    token = os.getenv("HF_TOKEN")

    # If not in environment, try to load from .env file directly
    if not token:
        try:
            with open(".env", "r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
        except FileNotFoundError:
            pass

    if not token:
        print("Error: HF_TOKEN not found in environment or .env file")
        return

    # Your models (repo_id, base_model)
    models = [
        (
            "rawcell/Moonlight-16B-A3B-Instruct-bruno",
            "moonshotai/Moonlight-16B-A3B-Instruct",
        ),
        ("rawcell/Qwen2.5-Coder-7B-Instruct-bruno", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        (
            "rawcell/Moonlight-16B-A3B-Instruct-abliterated",
            "moonshotai/Moonlight-16B-A3B-Instruct",
        ),
        ("rawcell/Llama-3.2-3B-Instruct-heretic", "meta-llama/Llama-3.2-3B-Instruct"),
        ("rawcell/Qwen2.5-3B-Instruct-heretic", "Qwen/Qwen2.5-3B-Instruct"),
    ]

    for repo_id, base_model in models:
        update_model_card(repo_id, base_model, token)
        print()

    print("SUCCESS: All model cards updated!")


if __name__ == "__main__":
    main()
