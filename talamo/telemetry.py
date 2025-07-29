"""Telemetry helpers for Talamo.

This module centralises OpenTelemetry instrumentation (traces) and Prometheus
metrics so that the router (and future components) can record latency and
expert selection without polluting business logic.
"""
from __future__ import annotations

import os
from typing import Final

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased  # type: ignore

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover – optional dependency
    _OTEL_AVAILABLE = False

try:
    from prometheus_client import Histogram, Counter  # type: ignore
except ImportError:  # pragma: no cover – optional dependency
    Histogram = Counter = None  # type: ignore

# ---------------------------------------------------------------------------
# OpenTelemetry tracer setup
# ---------------------------------------------------------------------------

SERVICE: Final = os.getenv("NEURON_SERVICE_NAME", "neuron-talamo")

if _OTEL_AVAILABLE:
    OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    SAMPLE_RATIO = float(os.getenv("OTEL_SAMPLE_RATIO", "0.1"))  # 10 % by default

    resource = Resource(attributes={SERVICE_NAME: SERVICE})
    provider = TracerProvider(
        resource=resource, sampler=ParentBased(TraceIdRatioBased(SAMPLE_RATIO))
    )
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(__name__)
else:
    # Fallback no-op tracer to keep code running without opentelemetry installed
    from contextlib import nullcontext

    class _NoOpTracer:  # noqa: D401
        """Minimal tracer replacement."""

        def start_as_current_span(self, *_a, **_kw):  # noqa: D401
            return nullcontext()

    tracer = _NoOpTracer()  # type: ignore
# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

if Histogram is not None:
    REQUEST_LATENCY_MS = Histogram(
        "talamo_router_latency_ms",
        "Latency of Talamo router route_input calls",
        buckets=(0.1, 0.5, 1, 2, 3, 4, 5, 10, 20),
    )

    EXPERT_COUNTER = Counter(
    "talamo_router_expert_total",
    "Count of expert chosen by Talamo router",
    labelnames=("expert",),
)


else:
    # Dummy placeholders
    REQUEST_LATENCY_MS = None  # type: ignore
    EXPERT_COUNTER = None  # type: ignore


def record_latency(latency_ms: float) -> None:  # noqa: D401
    """Record *latency_ms* in the Prometheus histogram."""

    if REQUEST_LATENCY_MS is not None:
        REQUEST_LATENCY_MS.observe(latency_ms)


def inc_expert(expert: str) -> None:  # noqa: D401
    """Increment Prometheus counter for *expert*."""

    if EXPERT_COUNTER is not None:
        EXPERT_COUNTER.labels(expert=expert).inc()
