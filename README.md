# TinyChat

A 561M-parameter language model trained from scratch for under $100 — tokenizer, pretraining, finetuning, deployment, and frontend.

**Try it live: [tinychat-two.vercel.app](https://tinychat-two.vercel.app)**

---

## What is this?

TinyChat is a fully custom ChatGPT-style chatbot built end-to-end. The model is a GPT-class transformer trained from raw text to conversational assistant in four stages, all on a single 8xH100 node. The total training cost was ~$95.

This project demonstrates the complete lifecycle of building an LLM product — from data processing and model training to serverless GPU deployment and a production web interface.

## Model

| | |
|---|---|
| **Parameters** | 561M (d20: 20 layers, 10 attention heads, 1280 embedding dim) |
| **Architecture** | GPT with RoPE, RMSNorm, Multi-Query Attention, ReLU-squared |
| **Tokenizer** | Custom BPE, 65K vocabulary |
| **Context window** | 2048 tokens |
| **Training data** | FineWeb-EDU (pretraining), SmolTalk + ARC + GSM8K + custom identity data (SFT) |
| **Tools** | Built-in calculator via special tokens for arithmetic |

## Training Pipeline

```
Custom BPE Tokenizer (65K vocab)
        |
        v
Base Pretraining (FineWeb-EDU, ~38B tokens)
        |
        v
Midtraining (instruction-following warmup)
        |
        v
Supervised Fine-Tuning (SmolTalk, ARC, GSM8K, identity data, spelling)
```

The identity data — teaching the model who it is and who built it — was generated synthetically (1,500 conversations via OpenRouter) and mixed into SFT at ~17% of the dataset.

## Deployment Stack

```
Browser  -->  Vercel (Next.js)  -->  Modal (T4 GPU, serverless)
                 |                        |
            API proxy route          FastAPI + SSE streaming
            (web/app/api/)           (nanochat/modal_app.py)
```

- **Frontend**: Next.js 16 on Vercel with Tailwind CSS 4
- **Backend**: Modal serverless GPU (NVIDIA T4) with 5-minute warm container window
- **Streaming**: Server-Sent Events from model inference through API proxy to browser
- **Auth**: API key header between Vercel and Modal

## Project Structure

```
.
├── nanochat/              # Training codebase (forked from karpathy/nanochat)
│   ├── nanochat/          # Core library (model, tokenizer, engine, inference)
│   ├── scripts/           # Training & serving scripts (base, mid, SFT, RL, web, CLI)
│   ├── tasks/             # Eval & SFT tasks (ARC, GSM8K, SmolTalk, custom JSON)
│   ├── dev/               # Synthetic data generation, dev utilities
│   ├── modal_app.py       # Modal deployment (inference server)
│   └── speedrun.sh        # End-to-end training script
│
└── web/                   # Next.js frontend
    ├── app/               # Pages + API route (chat proxy)
    └── components/        # Header, ChatMessage, ChatInput
```

## Based On

The training codebase is forked from [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat) — the capstone project for LLM101n. Customizations include synthetic identity data, the Modal deployment server, tool-token filtering, context window truncation, and the full web frontend.

See [nanochat/README.md](nanochat/README.md) for the original training documentation.

## License

MIT
