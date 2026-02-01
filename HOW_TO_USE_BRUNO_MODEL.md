# How to Use Your Bruno Model from HuggingFace

**Model:** https://huggingface.co/rawcell/bruno
**Size:** 65.5 GB (32B parameters)
**Status:** Fully uploaded and ready to use

---

## Option 1: Use on Your Laptop (Best Quality)

**Requirements:** RTX 4070 (8GB VRAM), 4-bit quantization needed

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load with 4-bit quantization (fits on 8GB VRAM)
model = AutoModelForCausalLM.from_pretrained(
    "rawcell/bruno",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16"
    )
)
tokenizer = AutoTokenizer.from_pretrained("rawcell/bruno")

# Chat with it
messages = [{"role": "user", "content": "Write a Python function to sort a list"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Option 2: Use from Phone (Inference API - FREE & EASY!)

### Method A: Python on Phone (Termux for Android)

**Install Termux from F-Droid, then:**
```bash
pkg install python
pip install huggingface_hub

python3
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient("rawcell/bruno")
>>> response = client.text_generation("Write Python code for sorting")
>>> print(response)
```

### Method B: HTTP API from Any Device

**Using curl (works everywhere):**
```bash
curl https://api-inference.huggingface.co/models/rawcell/bruno \
    -X POST \
    -H "Authorization: Bearer YOUR_HF_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"inputs": "Write code", "parameters": {"max_new_tokens": 512}}'
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "https://api-inference.huggingface.co/models/rawcell/bruno",
    headers={"Authorization": "Bearer YOUR_HF_TOKEN"},
    json={"inputs": "Write code", "parameters": {"max_new_tokens": 512}}
)
print(response.json())
```

### Method C: Web Interface (Easiest!)

1. Go to: https://huggingface.co/rawcell/bruno
2. Click "Deploy" → "Inference API"
3. Type your prompt in the web interface
4. Works from phone browser

**Note:** First inference may take 20-30 seconds to load model, then fast

---

## Option 3: Deploy as Web App (HuggingFace Spaces)

**Create a permanent chatbot:**

1. Go to: https://huggingface.co/new-space
2. Name: `bruno-chat`
3. SDK: Gradio
4. Create space

**Upload this app.py:**
```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("rawcell/bruno", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("rawcell/bruno")

def chat(message, history):
    messages = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    output = model.generate(inputs, max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

gr.ChatInterface(chat).launch()
```

**Access from:** Phone, tablet, laptop - anywhere
**URL:** https://huggingface.co/spaces/rawcell/bruno-chat

---

## Option 4: Use via OpenAI-Compatible API

**Deploy on HuggingFace Inference Endpoints:**

1. Go to: https://ui.endpoints.huggingface.co
2. Create endpoint for `rawcell/bruno`
3. Get API URL

**Then use like OpenAI:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-ENDPOINT.endpoints.huggingface.cloud/v1",
    api_key="YOUR_HF_TOKEN"
)

response = client.chat.completions.create(
    model="rawcell/bruno",
    messages=[{"role": "user", "content": "Write code"}]
)
print(response.choices[0].message.content)
```

**Works from phone using any OpenAI-compatible app!**

---

## Option 5: Download and Run Locally Later

**When you have time:**
```bash
# Download entire model
huggingface-cli download rawcell/bruno --local-dir ./models/bruno

# Or with Python
from huggingface_hub import snapshot_download
snapshot_download("rawcell/bruno", local_dir="./models/bruno")
```

**Then run with bruno:**
```bash
bruno ./models/bruno --n-trials 0  # Just chat, no more abliteration
```

---

## Option 6: Use on Cloud GPU (When Needed)

**Rent GPU and use from HuggingFace:**
```bash
# On cloud instance
pip install transformers torch

python3
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> model = AutoModelForCausalLM.from_pretrained("rawcell/bruno")
>>> # Use model
```

---

## Fastest Option for Phone: Inference API

**This works RIGHT NOW from your phone:**

1. Open browser on phone
2. Go to: https://huggingface.co/rawcell/bruno
3. Scroll to "Inference API" widget
4. Type prompt, click "Compute"
5. Get response in seconds

**Or use Python app on phone (Termux):**
```python
from huggingface_hub import InferenceClient

client = InferenceClient("rawcell/bruno")
result = client.text_generation("Write Python code for me", max_new_tokens=512)
print(result)
```

---

## Rate Limits (Free Tier)

**HuggingFace Inference API (free):**
- Requests: ~1000/day
- Speed: Fast after model loads
- First request: 20-30 sec (cold start)
- Subsequent: <5 sec

**If you need more:**
- Deploy on Inference Endpoints (~$0.60/hour when running)
- Or run locally when needed

---

## Summary

**Easiest (Phone/Anywhere):** HuggingFace web interface or Inference API
**Best Quality (Laptop):** Load with 4-bit quantization on RTX 4070
**24/7 Access (Phone):** Deploy HuggingFace Space with Gradio
**Professional (Apps):** Inference Endpoints with OpenAI-compatible API

**Your model is safe and accessible from anywhere at:**
https://huggingface.co/rawcell/bruno
