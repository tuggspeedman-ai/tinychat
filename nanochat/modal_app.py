"""
Modal deployment for TinyChat inference server.

Deploy with:
    modal deploy modal_app.py

Test locally with:
    modal serve modal_app.py

Endpoint will be available at:
    https://<your-username>--tinychat-chat-completions.modal.run
"""

import modal
import json
import os

# Create the Modal app
app = modal.App("tinychat")

# Secret for API key authentication
api_key_secret = modal.Secret.from_name("tinychat-api-key")

# Create a volume for model checkpoints
# Upload checkpoints with: modal volume put tinychat-checkpoints <local-path> <remote-path>
volume = modal.Volume.from_name("tinychat-checkpoints", create_if_missing=True)

# Build the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")  # For health checks
    .pip_install(
        # Core dependencies from pyproject.toml
        "torch>=2.0.0",
        "fastapi>=0.117.1",
        "uvicorn>=0.36.0",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "filelock>=3.0.0",
        "regex>=2025.9.1",
        # Note: We use tiktoken for inference, not rustbpe
        # rustbpe is only needed for training
    )
    # Copy the nanochat package into the image
    .add_local_dir("nanochat", remote_path="/app/nanochat", copy=True)
    .add_local_dir("scripts", remote_path="/app/scripts", copy=True)
    .env({
        "PYTHONPATH": "/app",
        "NANOCHAT_BASE_DIR": "/checkpoints"  # Tell nanochat where to find tokenizer
    })
)


# Global model cache to avoid reloading on warm requests
_model_cache = {}


def get_model(device):
    """Load model once and cache it for subsequent requests."""
    import torch
    from nanochat.checkpoint_manager import build_model

    if "model" not in _model_cache:
        print("Loading model from checkpoint...")

        # Checkpoints are mounted at /checkpoints from Modal volume
        checkpoint_dir = "/checkpoints/chatsft_checkpoints/d20"
        step = 809  # Phase 2 SFT checkpoint (TinyChat identity)

        model, tokenizer, meta_data = build_model(
            checkpoint_dir=checkpoint_dir,
            step=step,
            device=device,
            phase="eval"
        )

        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["meta_data"] = meta_data
        print(f"Model loaded successfully! Config: {meta_data.get('model_config', {})}")

    return _model_cache["model"], _model_cache["tokenizer"]


from typing import Annotated
from fastapi import Header
from pydantic import BaseModel

class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float = 0.8
    max_tokens: int = 512
    top_k: int = 50

@app.function(
    gpu="T4",
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[api_key_secret],
    scaledown_window=300,  # 5 minutes warm (adjust based on usage)
    timeout=120,  # Max 2 minutes per request
)
@modal.fastapi_endpoint(method="POST")
def chat_completions(request: ChatRequest, x_api_key: Annotated[str | None, Header()] = None):
    """
    TinyChat completions endpoint (SSE streaming).

    Request format:
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.8,
        "max_tokens": 512,
        "top_k": 50
    }

    Headers:
        X-API-Key: Required authentication key

    Returns: SSE stream with data: {"token": "...", "gpu": 0} chunks
    """
    import torch
    from fastapi.responses import StreamingResponse, JSONResponse
    from nanochat.engine import Engine
    import random

    # Check API key from header
    expected_key = os.environ.get("NANOCHAT_API_KEY")
    if not x_api_key or x_api_key != expected_key:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized - invalid or missing X-API-Key header"}
        )

    # Parse request (now using Pydantic model)
    messages = request.messages
    temperature = request.temperature
    max_tokens = request.max_tokens
    top_k = request.top_k

    # Validate
    if not messages:
        return {"error": "messages array is required"}

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (cached after first load)
    model, tokenizer = get_model(device)
    engine = Engine(model, tokenizer)

    # Build conversation tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    python_start = tokenizer.encode_special("<|python_start|>")
    python_end = tokenizer.encode_special("<|python_end|>")
    output_start = tokenizer.encode_special("<|output_start|>")
    output_end = tokenizer.encode_special("<|output_end|>")

    conversation_tokens = [bos]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(content))
            conversation_tokens.append(user_end)
        elif role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(content))
            conversation_tokens.append(assistant_end)

    # Prime for assistant response
    conversation_tokens.append(assistant_start)

    # Truncate old messages if conversation exceeds context window (2048 tokens)
    max_context = 2048
    max_prompt_tokens = max_context - max_tokens
    if len(conversation_tokens) > max_prompt_tokens:
        original_len = len(conversation_tokens)
        cut = original_len - max_prompt_tokens
        # Scan forward from cut point to find nearest message boundary (user_start)
        for i in range(cut, len(conversation_tokens)):
            if conversation_tokens[i] == user_start:
                conversation_tokens = [bos] + conversation_tokens[i:]
                break
        else:
            # Fallback: hard truncate, keeping BOS
            conversation_tokens = [bos] + conversation_tokens[-(max_prompt_tokens - 1):]
        print(f"Truncated conversation from {original_len} to {len(conversation_tokens)} tokens (limit: {max_prompt_tokens})")

    # Generate streaming response
    def generate():
        accumulated_tokens = []
        last_clean_text = ""
        in_python = False  # inside <|python_start|>...<|python_end|> (calculator expression)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for token_column, token_masks in engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=random.randint(0, 2**31 - 1)
            ):
                token = token_column[0]

                # Stopping criteria
                if token == assistant_end or token == bos:
                    break

                # Filter tool-use tokens: hide the machinery, keep the results
                if token == python_start:
                    in_python = True
                    continue
                if token == python_end:
                    in_python = False
                    continue
                if token in (output_start, output_end):
                    continue
                if in_python:
                    continue  # skip expression tokens (e.g. "12*4")

                # Accumulate tokens for proper UTF-8 handling
                accumulated_tokens.append(token)
                current_text = tokenizer.decode(accumulated_tokens)

                # Only emit text if it doesn't end with replacement character
                if not current_text.endswith('�'):
                    new_text = current_text[len(last_clean_text):]
                    if new_text:
                        yield f"data: {json.dumps({'token': new_text, 'gpu': 0}, ensure_ascii=False)}\n\n"
                        last_clean_text = current_text

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    import os

    # Check if checkpoint files exist
    checkpoint_dir = "/checkpoints/chatsft_checkpoints/d20"
    model_exists = os.path.exists(f"{checkpoint_dir}/model_000809.pt")
    meta_exists = os.path.exists(f"{checkpoint_dir}/meta_000809.json")
    tokenizer_exists = os.path.exists("/checkpoints/tokenizer/tokenizer.pkl")

    return {
        "status": "ok",
        "ready": model_exists and meta_exists and tokenizer_exists,
        "checkpoints": {
            "model": model_exists,
            "meta": meta_exists,
            "tokenizer": tokenizer_exists,
        }
    }


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test the deployment locally."""
    print("Testing health endpoint...")
    result = health.remote()
    print(f"Health: {result}")

    if result.get("ready"):
        print("\nTesting chat endpoint...")
        response = chat_completions.remote({
            "messages": [{"role": "user", "content": "Hello! Who are you?"}],
            "temperature": 0.8,
            "max_tokens": 100,
        })
        print(f"Response: {response}")
    else:
        print("\nCheckpoints not ready. Upload them first with:")
        print("  modal volume put tinychat-checkpoints ~/.cache/nanochat/chatsft_checkpoints /chatsft_checkpoints")
        print("  modal volume put tinychat-checkpoints ~/.cache/nanochat/tokenizer /tokenizer")
