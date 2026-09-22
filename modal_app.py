"""
Mission Control — a spacecraft/space-focused chatbot served with vLLM on Modal.

Architecture (one container, one GPU):
  - A Modal Image bundles CUDA + vLLM + FastAPI.
  - A Modal Volume caches Hugging Face model weights across restarts/deploys.
  - A Modal Cls loads the Qwen model once (on container start) into a vLLM
    engine, then serves both a JSON chat API and a static HTML UI through
    a single FastAPI app (`@modal.asgi_app`).
  - The system prompt keeps the bot scoped to spacecraft/space topics.

Deploy:
    modal deploy modal_app.py

Local dev loop (hot-reloads on save):
    modal serve modal_app.py

Steps 1-10 from the brief map onto this file as follows:
  1. Model choice        -> MODEL_NAME
  2-3. Modal account/CLI -> done once, outside this file (see README)
  4. Environment image    -> `image` below
  5. GPU selection        -> GPU_TYPE below
  6. Model caching        -> `hf_cache_vol` Modal Volume
  7. Start vLLM           -> AsyncLLMEngine in `Chatbot.load_model`
  8. Expose endpoint      -> `Chatbot.web` (@modal.asgi_app)
  9. Test                 -> see README "Test" section (curl / UI)
  10. Deploy              -> `modal deploy modal_app.py`
"""

import modal

# ----------------------------------------------------------------------------
# 1. Model choice
# ----------------------------------------------------------------------------
# Swap this for any open-weight instruct model vLLM supports (Qwen/Llama/Gemma
# family). Qwen2.5-7B-Instruct is a solid default: good quality, fits
# comfortably on a single 24GB GPU with room for KV cache.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "main"

# ----------------------------------------------------------------------------
# 5. GPU selection
# ----------------------------------------------------------------------------
# 7B model in bf16 is ~15GB of weights. An A10G/L4 (24GB) has enough headroom
# for weights + KV cache at moderate concurrency. Bump to "A100-40GB" (or
# tensor-parallel across multiple GPUs) for a bigger model or heavier traffic.
GPU_TYPE = "A10G"

SYSTEM_PROMPT = (
    "You are Mission Control, an assistant that answers questions about "
    "spacecraft, spaceflight, astronomy, and space exploration: rockets, "
    "satellites, orbital mechanics, crewed and uncrewed missions, space "
    "agencies, and related history and engineering. If a question falls "
    "clearly outside that scope, say so briefly and steer the conversation "
    "back to space topics. Be accurate, concise, and cite specific "
    "missions, spacecraft, or dates where it helps. If you are not sure "
    "of a fact, say so rather than guessing."
)

app = modal.App("space-chatbot")

# ----------------------------------------------------------------------------
# 4. Define the environment
# ----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3.post1",
        "fastapi[standard]==0.115.4",
        "huggingface_hub[hf_transfer]==0.26.2",
        "transformers==4.46.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ----------------------------------------------------------------------------
# 6. Add model caching
# ----------------------------------------------------------------------------
# Persist the Hugging Face cache so redeploys / container restarts don't
# re-download multi-gigabyte weights every time.
hf_cache_vol = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)
CACHE_PATH = "/cache/huggingface"


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={CACHE_PATH: hf_cache_vol},
    scaledown_window=60 * 10,   # keep the GPU warm for 10 min of idle time
    timeout=60 * 20,
    secrets=[],  # add modal.Secret.from_name(...) here for auth tokens etc.
)
@modal.concurrent(max_inputs=16)  # batch several chat requests per container
class Chatbot:
    @modal.enter()
    def load_model(self):
        """
        7. Start vLLM
        Runs once per container start. Loads the model into a vLLM async
        engine that stays resident for the life of the container.
        """
        import os

        os.environ["HF_HOME"] = CACHE_PATH

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            tensor_parallel_size=1,       # bump if using multiple GPUs
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    async def _generate(self, history: list[dict], max_tokens: int = 512):
        import uuid

        from vllm import SamplingParams

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        request_id = str(uuid.uuid4())
        final_output = None
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            final_output = output
        return final_output.outputs[0].text.strip()

    # ------------------------------------------------------------------
    # 8. Expose an endpoint
    # ------------------------------------------------------------------
    @modal.asgi_app()
    def web(self):
        from pathlib import Path

        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel

        web_app = FastAPI(title="Mission Control — Space Chatbot")

        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatRequest(BaseModel):
            messages: list[ChatMessage]

        class ChatResponse(BaseModel):
            reply: str

        @web_app.post("/api/chat", response_model=ChatResponse)
        async def chat(req: ChatRequest):
            if not req.messages:
                raise HTTPException(400, "messages must not be empty")
            history = [m.model_dump() for m in req.messages]
            reply = await self._generate(history)
            return ChatResponse(reply=reply)

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_NAME}

        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            web_app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
        else:
            @web_app.get("/", response_class=HTMLResponse)
            async def root():
                return "<h1>Mission Control</h1><p>static/index.html not found.</p>"

        return web_app
