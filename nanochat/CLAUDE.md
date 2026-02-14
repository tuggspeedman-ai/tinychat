# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack implementation of an LLM like ChatGPT in a minimal, hackable codebase. It trains models from scratch through the entire pipeline: tokenization, pretraining, midtraining, supervised finetuning (SFT), and optional reinforcement learning (RL). The goal is accessibility - models that can be trained end-to-end on budgets under $1000.

## Training Status (This Implementation)

**Phase 1 Complete (December 2024): Full Training Pipeline**

This repository has completed the full training pipeline for a d20 model (561M parameters):

- **Tokenizer**: 65,536 vocab, 4.8 chars/token compression
- **Base Pretraining**: 21,400 iterations, 11.2B tokens, ~2.86 hours on 8xH100
- **Midtraining**: 809 iterations, conversation format + tool use, ~6.3 minutes on 8xH100
- **Supervised Fine-tuning**: 700 iterations, 23K high-quality examples, ~7 minutes on 8xH100

**Total Training Cost**: ~$95 on Lambda Labs 8xH100 instances

**Final Model Performance (SFT stage)**:
- ARC-Easy: 46.46% (science reasoning)
- ARC-Challenge: 34.13% (advanced science)
- MMLU: 33.24% (multi-domain knowledge)
- GSM8K: 5.38% (grade school math)
- HumanEval: 10.98% (Python code generation)
- SpellingBee: 97.27% (character counting)

**Checkpoints Available** (organized in `../checkpoints/`):
- Base pretrained: [checkpoints/base/](../checkpoints/base/) (step 21400)
- Midtrained: [checkpoints/mid/](../checkpoints/mid/) (step 809)
- Phase 1 SFT: [checkpoints/sft/](../checkpoints/sft/) (step 700)
- Phase 2 SFT (TinyChat): [checkpoints/sft2/](../checkpoints/sft2/) (step 809, Feb 2025)
- Training data: [checkpoints/data/](../checkpoints/data/) (identity_conversations.jsonl)
- Tokenizer: [checkpoints/tokenizer/](../checkpoints/tokenizer/)

See [nanochat_project_plan.md](nanochat_project_plan.md) for complete Phase 1 training details and benchmarks.

**Phase 2 Complete: TinyChat Portfolio Project**

Renamed the model's user-facing identity to "TinyChat" and deployed it as a portfolio project for Jonathan Avni.

**Live at: https://tinychat-two.vercel.app**

- [x] Rewrite synthetic data generator to teach the model its new identity and about Jonathan
- [x] Rename user-facing strings from "nanochat" to "TinyChat" (Python module name stays `nanochat`)
- [x] Generate 1,498 synthetic identity conversations (1.3MB)
- [x] Re-run SFT with increased identity data proportion (~17% of mix) — 810 iterations on 8x A100, val_loss=1.014
- [x] Test and verify model identity responses via CLI
- [x] Build Next.js 16 chat frontend (Tailwind 4, dark theme, streaming SSE)
- [x] Deploy model to Modal (T4 GPU, serverless, SSE streaming)
- [x] Deploy frontend to Vercel + connect to Modal backend
- [x] Tool token filtering (hide `<|python_start|>` etc. from output, keep calculator results)
- [x] Context window truncation (drop oldest messages when conversation exceeds 2048 tokens)

**Phase 2 SFT checkpoint**: `checkpoints/sft2/` (step 809, Feb 2025). Use `--step 809` when loading via checkpoint manager since older checkpoints exist in `~/.cache/nanochat/chatsft_checkpoints/d20/`.

See [nanochat_project_plan_phase2.md](nanochat_project_plan_phase2.md) for the full Phase 2 plan.

## Environment Setup

This project uses `uv` for dependency management:

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (choose one):
uv sync --extra gpu    # For CUDA/GPU training
uv sync --extra cpu    # For CPU-only environments
```

The Rust tokenizer (`rustbpe`) must be built separately:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v -s

# Run specific test file
python -m pytest tests/test_rustbpe.py -v -s

# Skip slow tests
python -m pytest -m "not slow" -v -s
```

## Training Pipeline

The complete training pipeline runs through these stages:

### 1. Tokenizer Training
```bash
# Downloads ~800MB of training data and trains BPE tokenizer with vocab_size=65536
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval  # Evaluate compression ratio
```

### 2. Base Model Pretraining
```bash
# Single GPU (will use gradient accumulation)
python -m scripts.base_train -- --depth=20

# Multi-GPU distributed training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

Key training parameters:
- `--depth`: Model depth (number of Transformer layers). d20=561M params, d26=~GPT-2 scale, d32=1.9B params
- `--device_batch_size`: Per-device batch size (reduce if OOM, default 32)
- `--max_seq_len`: Context length (default 2048)
- `--run`: wandb run name (use "dummy" to skip wandb logging)

### 3. Midtraining (Conversation Format Training)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

Midtraining teaches the model conversation special tokens, tool use, and multiple choice format. Mixes pretraining data with task-specific datasets.

### 4. Supervised Finetuning (SFT)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

### 5. Reinforcement Learning (Optional)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

Currently only supports GSM8K task.

## Inference and Interaction

```bash
# Chat via CLI
python -m scripts.chat_cli -p "Why is the sky blue?"
python -m scripts.chat_cli  # Interactive mode (omit -p)

# Web UI (ChatGPT-style interface)
python -m scripts.chat_web
# Then visit http://localhost:8000 or http://<node-ip>:8000
```

## Complete Training Scripts

- `speedrun.sh`: ~$100 tier, d20 model (561M params), ~4 hours on 8XH100
- `run1000.sh`: ~$800 tier, d32 model (1.9B params), ~33 hours on 8XH100

To run in a screen session with logging:
```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# Detach with Ctrl-a d, reattach with: screen -r speedrun
```

## Architecture and Key Components

### Model Architecture (nanochat/gpt.py)
- GPT-style decoder-only Transformer
- Rotary embeddings (RoPE), no positional embeddings
- QK normalization in attention
- Untied weights (separate token embedding and lm_head)
- ReLU² activation in MLP
- RMSNorm without learnable parameters
- Multi-Query Attention (MQA) support for efficient inference

### Distributed Training
- Uses PyTorch DDP (DistributedDataParallel) via `torchrun`
- Custom distributed optimizers: `DistMuon` (for weight matrices) and `DistAdamW` (for embeddings)
- Training automatically switches to gradient accumulation on single GPU to maintain same effective batch size

### Data Pipeline (nanochat/dataloader.py)
- Streams text from parquet files in `~/.cache/nanochat/data/`
- Tokenizes on-the-fly with batching
- Supports approximate training resumption via state dicts
- Train/val split: all shards except last (train), last shard only (val)

### Checkpointing (nanochat/checkpoint_manager.py)
Checkpoints are saved to `~/.cache/nanochat/checkpoints/{model_tag}/` with structure:
- `base/`: Base pretrained model
- `mid/`: Midtrained model
- `sft/`: Supervised finetuned model
- `rl/`: RL-trained model

To load specific checkpoints:
```bash
python -m scripts.chat_cli --source=mid --step=500
```

### Tasks System (tasks/)
All evaluation tasks inherit from `Task` base class:
- `Task`: Base class with slicing support
- `TaskMixture`: Combines multiple tasks for training
- `TaskSequence`: Sequential task composition

Available tasks: ARC (Easy/Challenge), GSM8K, HumanEval, MMLU, SmolTalk, SpellingBee, CustomJSON

### Inference Engine (nanochat/engine.py)
- Efficient generation with KV-cache
- Handles conversation history and special tokens
- Built-in calculator tool support via `<calc>` tags
- Batched token generation

### Report Generation
Training runs automatically generate `report.md` with:
- System info and timestamps
- Evaluation metrics across all training stages (CORE, ARC, GSM8K, HumanEval, MMLU, ChatCORE)
- Model architecture and codebase statistics

## Customization

### TinyChat Identity (Phase 2)
The model is being customized as "TinyChat" for Jonathan Avni's portfolio. Key files:
- `dev/gen_synthetic_data.py`: Generates synthetic identity conversations via OpenRouter API
- `scripts/chat_sft.py`: SFT training mixture (identity data is `CustomJSON` task)
- Identity data is stored at `~/.cache/nanochat/identity_conversations.jsonl`
- User-facing name "TinyChat" appears in `nanochat/ui.html`, `scripts/chat_cli.py`, `scripts/chat_web.py`
- The Python module name remains `nanochat` (internal only, not renamed)

### Adding Custom Identity/Personality
See [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139). Generate synthetic conversations and mix into midtraining/SFT data using `CustomJSON` task.

Example in midtraining:
```python
identity_data = CustomJSON(get_base_dir() + "/identity_conversations.jsonl")
```

### Adding New Abilities
See [Guide: counting r in strawberry](https://github.com/karpathy/nanochat/discussions/164). Add new task-specific datasets and include in training mixture.

## Memory Management

If you encounter OOM errors:
- Reduce `--device_batch_size` (e.g., from 32 → 16 → 8 → 4 → 2 → 1)
- Code automatically compensates by increasing gradient accumulation steps
- Larger models (d26, d32) typically need smaller batch sizes

## Device Support

- CUDA (primary target): Full support with bfloat16 mixed precision
- CPU: Supported, use `dev/runcpu.sh` as reference for smaller configs
- MPS (Apple Silicon): Supported, auto-detected on MacBooks
- Device auto-detection: Leave `device_type=""` to auto-select CUDA > MPS > CPU

## Distributed Training Notes

- All scripts support both single-GPU and multi-GPU via `torchrun`
- Use `torchrun --standalone --nproc_per_node=N` where N is number of GPUs
- Omit `torchrun` for single GPU (results are identical, just slower)
- Code handles DDP rank 0 as master process for logging/checkpointing

## Configuration System (nanochat/configurator.py)

All scripts use a simple config override system:
```bash
# Override via CLI
python -m scripts.base_train -- --depth=26 --device_batch_size=16

# Or via config file
python -m scripts.base_train -- -c myconfig.txt
```

## WandB Integration

```bash
# Login once
wandb login

# Use named runs (instead of "dummy")
WANDB_RUN=my_run_name bash speedrun.sh
# or
python -m scripts.base_train -- --run=my_run_name
```

## Contributing Notes

From the project README:
- nanochat prioritizes simplicity and readability over exhaustive configurability
- No giant configuration objects, model factories, or complex abstraction layers
- Designed to be a "strong baseline" that is maximally forkable
- Current policy: disclose any substantial LLM contributions in PRs

## Data Storage

Default base directory: `~/.cache/nanochat/`
- `data/`: Pretraining parquet shards (~100MB each)
- `checkpoints/`: Model checkpoints
- `report/`: Training report sections (compiled into report.md)

Override with: `export NANOCHAT_BASE_DIR=/path/to/dir`

## Web Frontend

Next.js chat frontend at `../web/`:
- Next.js 16 + TypeScript + Tailwind 4
- Deployed on Vercel: https://tinychat-two.vercel.app
- API proxy route (`/api/chat`) keeps Modal API key server-side
- Streaming SSE with token-by-token display
- Suggestion buttons, auto-scroll, abort support
- Disclaimer: "TinyChat is a 561M parameter model. It will confidently hallucinate — that's the fun part."

## Deployment

### Modal (Model Backend)
- Config: `modal_app.py`
- Endpoint: `https://tuggspeedman-ai--tinychat-chat-completions.modal.run`
- GPU: T4 (serverless, 5-min scaledown window)
- Volume: `tinychat-checkpoints` (model_000809.pt + tokenizer)
- Secret: `tinychat-api-key` (API auth via X-API-Key header)
- Features: tool token filtering, context window truncation (2048 tokens)
- Deploy: `cd nanochat && modal deploy modal_app.py`
- Force cold start: `modal app stop tinychat`

### Vercel (Frontend)
- Source: `../web/` directory
- URL: https://tinychat-two.vercel.app
- Env vars: `MODAL_API_URL`, `MODAL_API_KEY` (set in Vercel dashboard)
- Deploy: `cd web && npx vercel --prod --yes`
