# Mission Control — Qwen-powered space chatbot on Modal

A small FastAPI chat app, backed by a Qwen model served with vLLM, deployed
as a single GPU container on Modal. Scoped to spacecraft/space questions via
its system prompt. "Manned" in the sense that it's a single attended
service you run and watch (logs, `modal app` dashboard) — not a
multi-agent or autonomous system.

## Project structure

```
space-chatbot/
├── modal_app.py       # everything: image, GPU, cache, vLLM engine, FastAPI app
├── static/
│   └── index.html     # chat UI (vanilla HTML/CSS/JS, no build step)
├── requirements.txt   # reference only — Modal installs deps remotely
└── README.md
```

## 1. Choose a model

Default is `Qwen/Qwen2.5-7B-Instruct` (set in `MODEL_NAME` at the top of
`modal_app.py`). It's Apache-2.0 licensed and fits a single 24GB GPU.

Swap in another open-weight instruct model if you prefer:
- Smaller/cheaper: `Qwen/Qwen2.5-3B-Instruct`
- Bigger/better: `Qwen/Qwen2.5-14B-Instruct` (needs more VRAM, see step 5)
- Other families: any Llama or Gemma instruct checkpoint vLLM supports —
  just check that family's license terms (Llama and Gemma both have usage
  terms beyond plain Apache/MIT).

## 2. Create a Modal account + authenticate the CLI

```bash
pip install modal
modal setup          # opens a browser to log in / create an account
```

## 3. Install the CLI

Already done by `pip install modal` above. Verify:

```bash
modal --version
```

## 4. Define the environment

Handled in `modal_app.py` — the `image` object builds a Debian-slim image
with vLLM, FastAPI, and Hugging Face libraries. Modal builds this remotely,
so you don't need CUDA or a GPU on your own machine.

## 5. Select a GPU

Set via `GPU_TYPE` in `modal_app.py` (default `"A10G"`, 24GB VRAM). Rule of
thumb for vLLM:

| Model size | Suggested GPU        |
|-----------|------------------------|
| 3B        | `"L4"` (24GB) or `"A10G"` |
| 7B        | `"A10G"` or `"L4"` (24GB) |
| 14B       | `"A100-40GB"`           |
| 32B+      | `"A100-80GB"` or multi-GPU (`tensor_parallel_size`) |

Leave headroom above the raw weight size for KV cache — `gpu_memory_utilization=0.90`
in the engine args tells vLLM how much of the GPU it's allowed to use.

## 6. Model caching

`hf_cache_vol`, a `modal.Volume`, is mounted at `/cache/huggingface` and set
as `HF_HOME`. The first request after a fresh deploy downloads the model
once; every later cold start reuses the cached weights instead of
re-downloading them.

## 7. Start vLLM

Handled in `Chatbot.load_model` (an `@modal.enter()` hook), which runs once
per container start and loads an `AsyncLLMEngine` that stays resident for
the container's life — no per-request model load.

## 8. Expose an endpoint

`Chatbot.web`, decorated with `@modal.asgi_app()`, serves:
- `GET /` — the chat UI (`static/index.html`)
- `POST /api/chat` — `{"messages": [{"role": "user", "content": "..."}]}` → `{"reply": "..."}`
- `GET /health` — basic liveness check

## 9. Test

Run in dev mode first (hot-reloads, tears down when you stop it):

```bash
modal serve modal_app.py
```

This prints a URL like `https://<you>--space-chatbot-chatbot-web-dev.modal.run`.
Then:

```bash
# Health check
curl https://<your-url>/health

# Chat request + rough latency
time curl -X POST https://<your-url>/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What was the Apollo 11 lunar module called?"}]}'
```

Open the URL in a browser to use the UI directly. Watch throughput, errors,
and GPU utilization live with:

```bash
modal app logs space-chatbot
```

(GPU memory/utilization also show in the Modal dashboard for the running app.)

## 10. Deploy

```bash
modal deploy modal_app.py
```

This creates a persistent endpoint (stable URL, survives your terminal
closing) at `https://<you>--space-chatbot-chatbot-web.modal.run`. Redeploy
any time you edit `modal_app.py` or `static/index.html` — cached weights in
the Volume mean redeploys don't re-download the model.

## Notes / next steps

- **Concurrency**: `@modal.concurrent(max_inputs=16)` lets one container
  batch multiple in-flight chat requests through vLLM's continuous
  batching. Raise it for more throughput per GPU, or add
  `min_containers=1` to `app.cls(...)` to avoid cold starts.
- **Auth**: this endpoint is public once deployed. Add a shared secret or
  Modal's built-in auth (`modal.Secret` + a header check in `/api/chat`)
  before sharing the URL widely.
- **Guardrails**: the space-only scoping is a system prompt, not a hard
  filter — good enough for a simple attended bot, not a security boundary.
