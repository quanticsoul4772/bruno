"""
Heretic Chat - A sophisticated chat interface for abliterated models
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Generator

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("heretic_chat")


# =============================================================================
# Custom Exceptions
# =============================================================================


class HereticError(Exception):
    """Base exception for Heretic Chat errors."""

    pass


class ModelNotFoundError(HereticError):
    """Raised when a model path does not exist or is invalid."""

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        self.message = message or f"Model not found at path: {path}"
        super().__init__(self.message)


class ModelValidationError(HereticError):
    """Raised when model files are missing or invalid."""

    def __init__(self, path: str, missing_files: list[str]) -> None:
        self.path = path
        self.missing_files = missing_files
        self.message = (
            f"Model at {path} is missing required files: {', '.join(missing_files)}"
        )
        super().__init__(self.message)


class ModelLoadError(HereticError):
    """Raised when a model fails to load."""

    def __init__(self, path: str, cause: Exception | None = None) -> None:
        self.path = path
        self.cause = cause
        self.message = f"Failed to load model from {path}"
        if cause:
            self.message += f": {cause}"
        super().__init__(self.message)


class CUDAOutOfMemoryError(HereticError):
    """Raised when GPU runs out of memory."""

    def __init__(self, required_gb: float | None = None) -> None:
        self.required_gb = required_gb
        self.message = "CUDA out of memory"
        if required_gb:
            self.message += f" (required ~{required_gb:.1f} GB)"
        self.message += ". Try closing other applications or using a smaller model."
        super().__init__(self.message)


class TokenizationError(HereticError):
    """Raised when tokenization fails."""

    def __init__(self, cause: Exception | None = None) -> None:
        self.cause = cause
        self.message = "Failed to tokenize input"
        if cause:
            self.message += f": {cause}"
        super().__init__(self.message)


class GenerationError(HereticError):
    """Raised when text generation fails."""

    def __init__(self, cause: Exception | None = None) -> None:
        self.cause = cause
        self.message = "Failed to generate response"
        if cause:
            self.message += f": {cause}"
        super().__init__(self.message)


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR: Path = Path("models")
CHAT_HISTORY_DIR: Path = Path("chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Required files for a valid model
MODEL_WEIGHT_PATTERNS: list[str] = [
    "model.safetensors",
    "pytorch_model.bin",
    "model-00001-of-*.safetensors",
    "pytorch_model-00001-of-*.bin",
]
TOKENIZER_FILES: list[str] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
]

# Available models (will be populated dynamically)
AVAILABLE_MODELS: dict[str, str] = {}


# =============================================================================
# GPU Memory Monitoring
# =============================================================================


def get_gpu_memory_info() -> dict[str, float] | None:
    """Get current GPU memory usage information.

    Returns:
        Dictionary with 'used_gb', 'total_gb', 'free_gb', and 'percent_used',
        or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        reserved_bytes = torch.cuda.memory_reserved(device)

        total_gb = total_bytes / (1024**3)
        used_gb = reserved_bytes / (1024**3)
        free_gb = (total_bytes - reserved_bytes) / (1024**3)
        percent_used = (reserved_bytes / total_bytes) * 100

        return {
            "used_gb": used_gb,
            "total_gb": total_gb,
            "free_gb": free_gb,
            "percent_used": percent_used,
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def format_gpu_memory_status() -> str:
    """Format GPU memory status for display in UI.

    Returns:
        Human-readable string describing GPU memory usage.
    """
    if not torch.cuda.is_available():
        return "CPU mode (no GPU)"

    info = get_gpu_memory_info()
    if info is None:
        return "GPU: Unknown"

    return f"GPU: {info['used_gb']:.1f}/{info['total_gb']:.1f} GB ({info['percent_used']:.0f}%)"


# =============================================================================
# Model Validation
# =============================================================================


def validate_model_files(model_path: Path) -> list[str]:
    """Validate that a model directory contains all required files.

    Args:
        model_path: Path to the model directory.

    Returns:
        List of missing file descriptions (empty if all files present).
    """
    missing: list[str] = []

    # Check for config.json
    if not (model_path / "config.json").exists():
        missing.append("config.json")

    # Check for model weights (at least one pattern must match)
    has_weights = False
    for pattern in MODEL_WEIGHT_PATTERNS:
        if "*" in pattern:
            # Glob pattern
            if list(model_path.glob(pattern)):
                has_weights = True
                break
        elif (model_path / pattern).exists():
            has_weights = True
            break

    if not has_weights:
        missing.append("model weights (model.safetensors or pytorch_model.bin)")

    # Check for tokenizer files (at least one must exist)
    has_tokenizer = any((model_path / f).exists() for f in TOKENIZER_FILES)
    if not has_tokenizer:
        missing.append("tokenizer files (tokenizer.json or tokenizer_config.json)")

    return missing


def discover_models() -> dict[str, str]:
    """Discover available models in the models directory.

    Returns:
        Dictionary mapping display names to model paths.
    """
    models: dict[str, str] = {}
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                # Validate model files
                missing = validate_model_files(model_dir)
                if missing:
                    logger.warning(
                        f"Skipping {model_dir.name}: missing {', '.join(missing)}"
                    )
                    continue

                # Extract a friendly name
                name = model_dir.name
                display_name = name.replace("-", " ").title()
                models[display_name] = str(model_dir)
                logger.debug(f"Found valid model: {display_name} at {model_dir}")
    return models


# =============================================================================
# Model Manager
# =============================================================================


class ModelManager:
    """Manages model loading, caching, and text generation."""

    def __init__(self) -> None:
        self.current_model_path: str | None = None
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._generation_error: Exception | None = None

    def load_model(self, model_path: str) -> None:
        """Load a model if not already loaded.

        Args:
            model_path: Path to the model directory.

        Raises:
            ModelNotFoundError: If the model path does not exist.
            ModelValidationError: If required model files are missing.
            CUDAOutOfMemoryError: If GPU runs out of memory.
            ModelLoadError: If the model fails to load for other reasons.
        """
        if self.current_model_path == model_path and self.model is not None:
            logger.debug(f"Model already loaded: {model_path}")
            return  # Already loaded

        path = Path(model_path)

        # Validate path exists
        if not path.exists():
            raise ModelNotFoundError(model_path)

        # Validate model files
        missing = validate_model_files(path)
        if missing:
            raise ModelValidationError(model_path, missing)

        # Unload previous model
        if self.model is not None:
            logger.info("Unloading previous model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            logger.debug(f"GPU memory after unload: {format_gpu_memory_status()}")

        logger.info(f"Loading model from {model_path}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            self.current_model_path = model_path
            logger.info(f"Model loaded on {self.device}")
            logger.info(f"Memory status: {format_gpu_memory_status()}")

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise CUDAOutOfMemoryError() from e

        except Exception as e:
            # Clean up partial load
            self.model = None
            self.tokenizer = None
            self.current_model_path = None
            torch.cuda.empty_cache()

            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                raise CUDAOutOfMemoryError() from e
            raise ModelLoadError(model_path, e) from e

    def generate_stream(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        """Generate a streaming response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = deterministic, >0 = more random).

        Yields:
            Partial response strings as they are generated.

        Raises:
            GenerationError: If no model is loaded or generation fails.
        """
        if self.model is None or self.tokenizer is None:
            raise GenerationError(Exception("No model loaded"))

        # Validate messages - Qwen's chat template fails on None content
        if not messages:
            raise TokenizationError(Exception("No messages to process"))

        # Ensure all message content is a string (Qwen's Jinja template fails on None)
        validated_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            # Convert None to empty string, ensure everything is str
            if content is None:
                content = ""
            validated_messages.append({"role": str(role), "content": str(content)})
        messages = validated_messages

        self._generation_error = None

        try:
            # Use tokenize=True directly for most reliable cross-model compatibility
            # This avoids issues where some tokenizers return different types
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # Handle different return types from apply_chat_template:
            # - BatchEncoding (most common with return_tensors="pt")
            # - torch.Tensor (some tokenizers)
            # - list of ints (if return_tensors not supported)
            if hasattr(result, "input_ids"):
                # BatchEncoding or similar dict-like object
                input_ids = result.input_ids
                attention_mask = result.get("attention_mask", None)
            elif isinstance(result, torch.Tensor):
                input_ids = result
                attention_mask = None
            elif isinstance(result, list):
                if not result:
                    raise TokenizationError(Exception("Tokenizer returned empty list"))
                # List of token IDs
                input_ids = torch.tensor(
                    [result] if isinstance(result[0], int) else result
                )
                attention_mask = None
            else:
                raise TokenizationError(
                    Exception(f"Unexpected tokenizer output type: {type(result)}")
                )

            input_ids = input_ids.to(self.model.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(self.model.device)

        except TokenizationError:
            raise
        except Exception as e:
            logger.error(f"Tokenization failed: {type(e).__name__}: {e}")
            raise TokenizationError(e) from e

        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation config
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        # Run generation in a thread with error capture
        def generate_with_error_capture() -> None:
            try:
                self.model.generate(**gen_kwargs)  # type: ignore[union-attr]
            except Exception as e:
                self._generation_error = e
                logger.error(f"Generation error in thread: {e}")

        thread = Thread(target=generate_with_error_capture)
        thread.start()

        # Yield tokens as they arrive
        generated_text = ""
        try:
            for new_text in streamer:
                generated_text += new_text
                yield generated_text
        except Exception as e:
            thread.join()
            raise GenerationError(e) from e

        thread.join()

        # Check if generation thread had an error
        if self._generation_error is not None:
            raise GenerationError(self._generation_error)


# Global model manager
model_manager = ModelManager()


# =============================================================================
# Chat History Management
# =============================================================================


def save_chat_history(
    history: list[dict[str, str]], filename: str | None = None
) -> str:
    """Save chat history to a JSON file.

    Args:
        history: List of message dictionaries.
        filename: Optional filename (auto-generated if not provided).

    Returns:
        Status message indicating success or failure.
    """
    if not history:
        return "No history to save."

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"

    filepath = CHAT_HISTORY_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "history": history},
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Chat history saved to {filepath}")
        return f"Saved to {filepath}"
    except OSError as e:
        logger.error(f"Failed to save chat history: {e}")
        return f"Failed to save: {e}"


def load_chat_history(filename: str) -> tuple[list[dict[str, str]], str]:
    """Load chat history from a JSON file.

    Args:
        filename: Name of the file to load.

    Returns:
        Tuple of (history list, status message).
    """
    if not filename:
        return [], "Please select a chat to load."

    filepath = CHAT_HISTORY_DIR / filename
    if not filepath.exists():
        logger.warning(f"Chat history file not found: {filename}")
        return [], f"File not found: {filename}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Chat history loaded from {filepath}")
        return data.get("history", []), f"Loaded {filename}"
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load chat history: {e}")
        return [], f"Failed to load: {e}"


def get_saved_chats() -> list[str]:
    """Get list of saved chat files.

    Returns:
        List of filenames sorted by most recent first.
    """
    files = list(CHAT_HISTORY_DIR.glob("*.json"))
    return [f.name for f in sorted(files, reverse=True)]


# =============================================================================
# Chat Response Handler
# =============================================================================


def chat_response(
    message: str,
    history: list[dict[str, str]],
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Generate a chat response with streaming.

    Args:
        message: The user's message.
        history: Previous conversation history.
        model_name: Display name of the model to use.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Yields:
        Partial response strings as they are generated.
    """
    if not model_name or model_name not in AVAILABLE_MODELS:
        yield "Please select a model first."
        return

    # Load model if needed
    model_path = AVAILABLE_MODELS[model_name]
    try:
        model_manager.load_model(model_path)
    except CUDAOutOfMemoryError as e:
        logger.error(f"CUDA OOM: {e}")
        yield f"Error: {e.message}"
        return
    except ModelValidationError as e:
        logger.error(f"Model validation failed: {e}")
        yield f"Error: {e.message}"
        return
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e}")
        yield f"Error: {e.message}"
        return
    except ModelLoadError as e:
        logger.error(f"Model load failed: {e}")
        yield f"Error: {e.message}"
        return

    # Build messages list from history (Gradio 6 format: list of dicts)
    messages: list[dict[str, str]] = []
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
    except TokenizationError as e:
        logger.error(f"Tokenization error: {e}")
        yield f"Error tokenizing input: {e.cause}"
    except GenerationError as e:
        logger.error(f"Generation error: {e}")
        yield f"Error generating response: {e.cause}"


# =============================================================================
# Gradio UI
# =============================================================================


def create_ui() -> gr.Blocks:
    """Create the Gradio interface.

    Returns:
        Configured Gradio Blocks interface.
    """
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = discover_models()

    with gr.Blocks() as demo:
        # Header
        gr.HTML("""
            <div class="chat-header">
                <h1>Heretic</h1>
                <p>Uncensored local AI</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
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

            with gr.Column(scale=1):
                # GPU memory monitor
                gpu_status = gr.Textbox(
                    value=format_gpu_memory_status(),
                    label="GPU Memory",
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
        def user_message(
            user_msg: str, history: list[dict[str, str]] | None
        ) -> tuple[str, list[dict[str, str]]]:
            """Add user message to history."""
            if history is None:
                history = []
            history = history + [{"role": "user", "content": user_msg}]
            return "", history

        def bot_response(
            history: list[dict[str, str]], model_name: str, max_tok: int, temp: float
        ) -> Generator[list[dict[str, str]], None, None]:
            """Generate bot response."""
            if not history:
                return

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

        def update_status(model_name: str) -> str:
            """Update status when model changes."""
            if model_name and model_name in AVAILABLE_MODELS:
                return f"Loading {model_name}..."
            return "No model selected"

        def on_model_loaded(model_name: str) -> tuple[str, str]:
            """Called after model is used. Returns updated status and GPU info."""
            return f"[OK] {model_name} ready", format_gpu_memory_status()

        # Wire up events
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, model_dropdown, max_tokens, temperature], chatbot
        ).then(on_model_loaded, [model_dropdown], [status_text, gpu_status])

        submit_btn.click(
            user_message, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_response, [chatbot, model_dropdown, max_tokens, temperature], chatbot
        ).then(on_model_loaded, [model_dropdown], [status_text, gpu_status])

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


def main() -> None:
    """Main entry point."""
    logger.info("Heretic Chat - Starting...")
    logger.info(f"Models directory: {MODELS_DIR.absolute()}")
    logger.info(f"Chat history directory: {CHAT_HISTORY_DIR.absolute()}")

    # Discover models
    models = discover_models()
    if not models:
        logger.warning(
            "No models found! Please ensure models are in the 'models' directory."
        )
        logger.warning("   Expected structure: models/<model-name>/config.json")
    else:
        logger.info(f"Found {len(models)} model(s):")
        for name, path in models.items():
            logger.info(f"   - {name}: {path}")

    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {device_name}")
        logger.info(f"Initial GPU status: {format_gpu_memory_status()}")
    else:
        logger.warning("CUDA not available, using CPU (will be slower)")

    # Custom CSS for clean minimal styling
    custom_css = """
    .gradio-container {
        max-width: 850px !important;
        margin: auto !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif !important;
    }
    .chat-header {
        text-align: center;
        padding: 24px 20px 16px 20px;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 16px;
    }
    .chat-header h1 {
        color: #171717;
        margin: 0;
        font-size: 1.75em;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .chat-header p {
        color: #737373;
        margin: 6px 0 0 0;
        font-size: 0.9em;
        font-weight: 400;
    }
    footer {
        display: none !important;
    }
    /* Chat container styling */
    .chatbot {
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
    }
    /* Message styling */
    .message {
        border-radius: 6px !important;
    }
    /* Input styling */
    .input-container textarea {
        border: 1px solid #d4d4d4 !important;
        border-radius: 6px !important;
    }
    .input-container textarea:focus {
        border-color: #a3a3a3 !important;
        box-shadow: none !important;
    }
    /* Button styling */
    .primary {
        background: #171717 !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .primary:hover {
        background: #404040 !important;
    }
    /* Accordion styling */
    .accordion {
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
    }
    /* Remove extra borders and shadows */
    .block {
        box-shadow: none !important;
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
        theme=gr.themes.Monochrome(),
    )


if __name__ == "__main__":
    main()
