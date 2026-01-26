"""
Heretic Chat - A sophisticated chat interface for abliterated models
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Configuration
MODELS_DIR = Path("models")
CHAT_HISTORY_DIR = Path("chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Available models (will be populated dynamically)
AVAILABLE_MODELS = {}


def discover_models():
    """Discover available models in the models directory."""
    models = {}
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                # Extract a friendly name
                name = model_dir.name
                display_name = name.replace("-", " ").title()
                models[display_name] = str(model_dir)
    return models


# Model cache to avoid reloading
class ModelManager:
    def __init__(self):
        self.current_model_path = None
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str):
        """Load a model if not already loaded."""
        if self.current_model_path == model_path and self.model is not None:
            return  # Already loaded

        # Unload previous model
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.current_model_path = model_path
        print(f"Model loaded on {self.device}!")

    def generate_stream(
        self, messages: list, max_tokens: int = 512, temperature: float = 0.7
    ):
        """Generate a streaming response."""
        if self.model is None:
            yield "Error: No model loaded. Please select a model."
            return

        # Apply chat template - use tokenize=False first, then tokenize separately
        # This avoids issues with different return types from apply_chat_template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the prompt
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation config
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        # Run generation in a thread
        thread = Thread(target=lambda: self.model.generate(**gen_kwargs))
        thread.start()

        # Yield tokens as they arrive
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

        thread.join()


# Global model manager
model_manager = ModelManager()


def save_chat_history(history: list, filename: str = None):
    """Save chat history to a JSON file."""
    if not history:
        return "No history to save."

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"

    filepath = CHAT_HISTORY_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "history": history},
            f,
            indent=2,
            ensure_ascii=False,
        )

    return f"Saved to {filepath}"


def load_chat_history(filename: str):
    """Load chat history from a JSON file."""
    if not filename:
        return [], "Please select a chat to load."

    filepath = CHAT_HISTORY_DIR / filename
    if not filepath.exists():
        return [], f"File not found: {filename}"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("history", []), f"Loaded {filename}"


def get_saved_chats():
    """Get list of saved chat files."""
    files = list(CHAT_HISTORY_DIR.glob("*.json"))
    return [f.name for f in sorted(files, reverse=True)]


def chat_response(
    message: str, history: list, model_name: str, max_tokens: int, temperature: float
):
    """Generate a chat response with streaming."""
    if not model_name or model_name not in AVAILABLE_MODELS:
        yield "Please select a model first."
        return

    # Load model if needed
    model_path = AVAILABLE_MODELS[model_name]
    try:
        model_manager.load_model(model_path)
    except Exception as e:
        yield f"Error loading model: {str(e)}"
        return

    # Build messages list from history (Gradio 6 format: list of dicts)
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Generate streaming response
    try:
        for partial_response in model_manager.generate_stream(
            messages, max_tokens, temperature
        ):
            yield partial_response
    except Exception as e:
        yield f"Error generating response: {str(e)}"


def create_ui():
    """Create the Gradio interface."""
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = discover_models()

    with gr.Blocks() as demo:
        # Header
        gr.HTML("""
            <div class="chat-header">
                <h1>Heretic Chat</h1>
                <p>Uncensored AI - Powered by abliterated models</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0]
                    if AVAILABLE_MODELS
                    else None,
                    label="Select Model",
                    info="Choose which abliterated model to chat with",
                )

            with gr.Column(scale=1):
                # Status indicator
                status_text = gr.Textbox(
                    value="Ready" if AVAILABLE_MODELS else "No models found",
                    label="Status",
                    interactive=False,
                )

        # Chat interface - Gradio 6 uses messages format by default
        chatbot = gr.Chatbot(label="Chat", height=500)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here... (Press Enter to send)",
                label="Message",
                scale=4,
                show_label=False,
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")

        # Advanced settings (collapsed by default)
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Max Tokens",
                    info="Maximum length of the response",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused",
                )

        # Chat history management
        with gr.Accordion("Chat History", open=False):
            with gr.Row():
                save_btn = gr.Button("Save Chat", size="sm")
                save_status = gr.Textbox(
                    label="Save Status", interactive=False, scale=2
                )

            with gr.Row():
                history_dropdown = gr.Dropdown(
                    choices=get_saved_chats(), label="Load Previous Chat", scale=2
                )
                load_btn = gr.Button("Load", size="sm")
                refresh_btn = gr.Button("Refresh", size="sm")

            clear_btn = gr.Button("Clear Chat", variant="stop", size="sm")

        # Example prompts
        gr.Examples(
            examples=[
                "Tell me a creative story about a time traveler.",
                "Explain quantum computing like I'm 5 years old.",
                "Write a poem about the beauty of chaos.",
                "What are the ethical implications of AI development?",
                "Help me brainstorm ideas for a sci-fi novel.",
            ],
            inputs=msg,
            label="Try these prompts",
        )

        # Event handlers for Gradio 6 messages format
        def user_message(user_msg, history):
            """Add user message to history."""
            if history is None:
                history = []
            history = history + [{"role": "user", "content": user_msg}]
            return "", history

        def bot_response(history, model_name, max_tok, temp):
            """Generate bot response."""
            if not history:
                return history

            # Get the last user message
            user_msg = history[-1]["content"]

            # Add placeholder for assistant response
            history = history + [{"role": "assistant", "content": ""}]

            # Generate response
            for response in chat_response(
                user_msg, history[:-2], model_name, max_tok, temp
            ):
                history[-1]["content"] = response
                yield history

        def update_status(model_name):
            """Update status when model changes."""
            if model_name and model_name in AVAILABLE_MODELS:
                return f"Loading {model_name}..."
            return "No model selected"

        def on_model_loaded(model_name):
            """Called after model is used."""
            return f"[OK] {model_name} ready"

        # Wire up events
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, model_dropdown, max_tokens, temperature], chatbot
        ).then(on_model_loaded, [model_dropdown], [status_text])

        submit_btn.click(
            user_message, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_response, [chatbot, model_dropdown, max_tokens, temperature], chatbot
        ).then(on_model_loaded, [model_dropdown], [status_text])

        model_dropdown.change(update_status, [model_dropdown], [status_text])

        save_btn.click(lambda h: save_chat_history(h), [chatbot], [save_status])

        load_btn.click(
            lambda f: load_chat_history(f), [history_dropdown], [chatbot, save_status]
        )

        refresh_btn.click(
            lambda: gr.update(choices=get_saved_chats()), outputs=[history_dropdown]
        )

        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


def main():
    """Main entry point."""
    print("Heretic Chat - Starting...")
    print(f"Models directory: {MODELS_DIR.absolute()}")
    print(f"Chat history directory: {CHAT_HISTORY_DIR.absolute()}")

    # Discover models
    models = discover_models()
    if not models:
        print(
            "WARNING: No models found! Please ensure models are in the 'models' directory."
        )
        print("   Expected structure: models/<model-name>/config.json")
    else:
        print(f"Found {len(models)} model(s):")
        for name, path in models.items():
            print(f"   - {name}: {path}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, using CPU (will be slower)")

    # Custom CSS and theme for launch
    custom_css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    .chat-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .chat-header h1 {
        color: #e94560;
        margin: 0;
        font-size: 2.5em;
    }
    .chat-header p {
        color: #a0a0a0;
        margin: 10px 0 0 0;
    }
    footer {
        display: none !important;
    }
    """

    # Create and launch UI
    demo = create_ui()
    demo.queue()  # Enable queuing for streaming
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to get a public URL
        inbrowser=True,
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
