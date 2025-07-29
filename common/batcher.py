"""Generic asyncio batcher utility.

Collects calls to a function and executes them in batches of up to `batch_size`,
with a small timeout (`max_wait_ms`) to trade latency for throughput.

Example:
    from common.batcher import AsyncBatcher

    async_gen = AsyncBatcher(generate_fn, batch_size=8)
    result = await async_gen("Hello")
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, List


class AsyncBatcher:
    """Batch async calls to *func*.

    func should accept a `List[Any]` of inputs and return a List of outputs in
    the same order.
    """

    def __init__(self, func: Callable[[List[Any]], Awaitable[List[Any]]], batch_size: int = 8, max_wait_ms: int = 5):
        self._func = func
        self._batch_size = batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[tuple[Any, asyncio.Future]] = asyncio.Queue()
        self._task = asyncio.create_task(self._runner())

    async def _runner(self) -> None:
        while True:
            inputs: List[Any] = []
            futures: List[asyncio.Future] = []
            # Wait for at least one item
            inp, fut = await self._queue.get()
            inputs.append(inp)
            futures.append(fut)
            start = time.perf_counter()
            # Gather until batch full or timeout
            while len(inputs) < self._batch_size and (time.perf_counter() - start) < self._max_wait:
                try:
                    inp2, fut2 = self._queue.get_nowait()
                    inputs.append(inp2)
                    futures.append(fut2)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)  # yield
            # Execute
            try:
                outputs = await self._func(inputs)
            except Exception as exc:  # noqa: BLE001
                for fut in futures:
                    fut.set_exception(exc)
            else:
                for fut, out in zip(futures, outputs):
                    fut.set_result(out)

    async def __call__(self, inp: Any) -> Any:  # noqa: D401
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((inp, fut))
        return await fut
