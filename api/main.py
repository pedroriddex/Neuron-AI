"""FastAPI Gateway `/chat` integrating Talamo, Wernicke, Broca.

Usage (local):
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from talamo import TalamoRouter, Expert
import asyncio
from functools import partial

from broca import generate as broca_generate
from common.batcher import AsyncBatcher
from wernicke import infer as wernicke_infer

app = FastAPI(title="Neuron MVP Chat API", version="0.1.0")

router = TalamoRouter()

# Batcher for generation (up to 8 req, 5 ms window)
async def _gen_batch(prompts):  # List[str] -> List[str]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: [broca_generate(p) for p in prompts])

gen_batcher = AsyncBatcher(_gen_batch, batch_size=8, max_wait_ms=5)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096, description="User prompt")


class ChatResponse(BaseModel):
    response: str
    expert: str
    latency_ms: float


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):  # noqa: D401
    t0 = time.perf_counter_ns()

    if len(req.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # 1. Routing (currently single expert)
    experts: List[Expert] = router.route_input(req.prompt)
    expert = experts[0]

    # 2. Process according to expert (simplified MVP logic)
    if expert in (Expert.WERNICKE, Expert.GENERIC):
        _ = wernicke_infer([req.prompt])  # comprehension side-effect (ignored)
    # Broca always generates final text
    response_text = await gen_batcher(req.prompt)

    latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
    return ChatResponse(response=response_text, expert=expert.name, latency_ms=latency_ms)


# Health endpoint
@app.get("/healthz")
async def health() -> dict[str, str]:  # noqa: D401
    return {"status": "ok"}
