"""Stub implementation of TalamoRouter (Sprint-0).

This minimal version hard-codes three experts and uses a simple
round-robin strategy to choose an expert for each incoming request.
The method `route_input` is intentionally lightweight so that the call
returns in <5 ms on commodity hardware.

Telemetry instrumentation will be added in T-005.
"""
from __future__ import annotations

import itertools
from talamo.telemetry import tracer, record_latency, inc_expert
import time
from enum import Enum, auto
from typing import List

__all__ = ["Expert", "TalamoRouter"]


class Expert(Enum):
    """Enumeration of hard-coded experts for the MVP stub."""

    WERNICKE = auto()
    BROCA = auto()
    GENERIC = auto()


class TalamoRouter:
    """Minimal Expert-Choice router (stub).

    This version keeps an internal iterator cycling through the
    available experts. It returns the selected expert and the elapsed
    routing time (for tests/metrics).
    """

    _cycle_iter = itertools.cycle(list(Expert))

    def __init__(self) -> None:
        self._last_chosen: Expert | None = None

    def route_input(self, user_input: str) -> List[Expert]:
        """Select an expert for the given *user_input*.

        Parameters
        ----------
        user_input:
            Raw user utterance. Currently unused by the stub.

        Returns
        -------
        List[Expert]
            A list with a single chosen expert.
        """
        with tracer.start_as_current_span("talamo.route_input") as span:
            start = time.perf_counter_ns()

            # Simple round-robin selection ─ deterministic & O(1)
            chosen = next(self._cycle_iter)
            self._last_chosen = chosen

            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            # Telemetry
            if hasattr(span, "set_attribute"):
                span.set_attribute("expert", chosen.name)
                span.set_attribute("latency_ms", elapsed_ms)
            record_latency(elapsed_ms)
            inc_expert(chosen.name)

        # Assert latency (<5 ms) in debug mode
        assert elapsed_ms < 5, (
            "TalamoRouter routing stub took too long: " f"{elapsed_ms:.3f} ms"
        )

        return [chosen]
