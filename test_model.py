import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Llama-3.2-3B-heretic model...")
model_path = "models/llama-3.2-3b-heretic"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

print(f"Model loaded on {model.device}!")
print("-" * 50)

# Test with a creative writing prompt
messages = [{"role": "user", "content": "Write a short fictional story about a heist where the main character breaks into a museum."}]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
if not isinstance(input_ids, torch.Tensor):
    input_ids = input_ids["input_ids"]
input_ids = input_ids.to(model.device)

print("Generating response...\n")
with torch.no_grad():
    outputs = model.generate(
        input_ids, 
        max_new_tokens=300, 
        temperature=0.7, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print("PROMPT: Write a short fictional story about a heist where the main character breaks into a museum.")
print("-" * 50)
print("RESPONSE:")
print(response)
