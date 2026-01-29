"""
Test Qwen baseline verbosity on coding prompts.
This helps us understand how verbose the model is before abliteration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Coding prompts to test - mix of simple and complex
CODING_PROMPTS = [
    "Write a Python function to reverse a string",
    "Write a JavaScript function to check if a number is prime",
    "Write a Python function to find the factorial of a number",
    "Write a function to merge two sorted arrays",
    "Write a simple bubble sort implementation in Python",
    "Write a function to check if a string is a palindrome",
    "Write a Python function to calculate fibonacci numbers",
    "Write a function to remove duplicates from a list",
]

# Markers that indicate verbose explanations (not just code)
VERBOSE_MARKERS = [
    "Here's",
    "Here is",
    "This function",
    "This code",
    "This will",
    "Let me",
    "I'll",
    "I've",
    "The following",
    "This implementation",
    "explanation",
    "Explanation",
    "works by",
    "Here's how",
    "Let's break",
    "Step by step",
    "First,",
    "Note:",
    "Note that",
]


def load_model(model_path: str):
    """Load model in 4-bit for testing on consumer GPU."""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 500) -> str:
    """Generate a response for a coding prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def analyze_response(response: str) -> dict:
    """Analyze a response for verbosity markers."""
    # Count markers found
    markers_found = []
    for marker in VERBOSE_MARKERS:
        if marker.lower() in response.lower():
            markers_found.append(marker)
    
    # Count lines that are code vs prose
    lines = response.strip().split('\n')
    code_lines = 0
    prose_lines = 0
    
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block or stripped.startswith('def ') or stripped.startswith('function ') or stripped.startswith('    ') or stripped.startswith('\t'):
            code_lines += 1
        elif stripped:
            prose_lines += 1
    
    return {
        "total_chars": len(response),
        "total_lines": len(lines),
        "code_lines": code_lines,
        "prose_lines": prose_lines,
        "markers_found": markers_found,
        "marker_count": len(markers_found),
        "prose_ratio": prose_lines / max(len(lines), 1),
    }


def main():
    # Test with the original Qwen model
    model_path = "./models/Qwen2.5-7B-Instruct"
    
    print("=" * 60)
    print("BASELINE VERBOSITY TEST - Qwen2.5-7B-Instruct")
    print("=" * 60)
    
    model, tokenizer = load_model(model_path)
    
    results = []
    
    for prompt in CODING_PROMPTS:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print("=" * 60)
        
        response = generate_response(model, tokenizer, prompt)
        analysis = analyze_response(response)
        
        print(f"\nRESPONSE ({analysis['total_chars']} chars, {analysis['total_lines']} lines):")
        print("-" * 40)
        print(response[:1000] + ("..." if len(response) > 1000 else ""))
        print("-" * 40)
        print(f"Code lines: {analysis['code_lines']}")
        print(f"Prose lines: {analysis['prose_lines']}")
        print(f"Prose ratio: {analysis['prose_ratio']:.1%}")
        print(f"Verbose markers found ({analysis['marker_count']}): {analysis['markers_found']}")
        
        results.append({
            "prompt": prompt,
            "response": response,
            **analysis
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_chars = sum(r["total_chars"] for r in results) / len(results)
    avg_prose_ratio = sum(r["prose_ratio"] for r in results) / len(results)
    avg_markers = sum(r["marker_count"] for r in results) / len(results)
    
    print(f"Average response length: {avg_chars:.0f} chars")
    print(f"Average prose ratio: {avg_prose_ratio:.1%}")
    print(f"Average verbose markers per response: {avg_markers:.1f}")
    
    # Count which markers appear most
    all_markers = []
    for r in results:
        all_markers.extend(r["markers_found"])
    
    from collections import Counter
    marker_counts = Counter(all_markers)
    print(f"\nMost common verbose markers:")
    for marker, count in marker_counts.most_common(10):
        print(f"  '{marker}': {count} times")


if __name__ == "__main__":
    main()
