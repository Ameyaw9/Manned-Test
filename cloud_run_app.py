import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are manned, a thoughtful and capable AI assistant. Be clear, warm, concise, and honest about uncertainty.",
)

app = FastAPI(title="manned", version="1.0.0")
engine = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 512

@app.on_event("startup")
async def load_model():
    global engine, tokenizer
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
        dtype=os.getenv("MODEL_DTYPE", "bfloat16"),
    ))

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model is loading")
    from vllm import SamplingParams
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [m.model_dump() for m in req.messages]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = None
    async for result in engine.generate(prompt, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=min(req.max_tokens, 2048)), str(uuid.uuid4())):
        output = result
    return {"reply": output.outputs[0].text.strip() if output else ""}

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")
