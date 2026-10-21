"""Observe local backend work only in explicitly isolated acceptance sessions.

No prompts, source strings, generated strings or credentials enter these events.
Context variables prevent concurrent preload/prefetch work from being attributed
to a screen frame. Ordinary personal sessions do not collect or persist traces.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from .acceptance_evidence import evidence_enabled, save_execution_event

PROTOCOL = "pingyitong-native-stages-v2"


@dataclass
class ExecutionTrace:
    seq: int
    epoch: int | None
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=time.perf_counter)
    finished_at: float | None = None
    completed: bool = False
    events: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "protocol_version": PROTOCOL,
            "trace_id": self.trace_id,
            "seq": self.seq,
            "translation_epoch": self.epoch,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "completed": self.completed,
            "coverage": "model_pool_local_and_api_calls",
            "counts": {kind: sum(event["kind"] == kind for event in self.events)
                       for kind in ("model_load", "model_warmup", "local_inference", "api_inference")},
            "events": [dict(event) for event in self.events],
        }


_trace: ContextVar[ExecutionTrace | None] = ContextVar("translation_execution_trace", default=None)


@contextmanager
def trace_translation(frame):
    if not evidence_enabled():
        yield None
        return
    trace = ExecutionTrace(frame.seq, getattr(frame, "_translation_epoch", None))
    token = _trace.set(trace)
    try:
        yield trace
        trace.completed = True
    finally:
        trace.finished_at = time.perf_counter()
        _trace.reset(token)


@contextmanager
def model_work(kind: str, model_id: str):
    """Time an actual backend operation, including failed/raising operations."""
    trace = _trace.get()
    if trace is None and not evidence_enabled():
        yield None
        return
    event = {
        "kind": kind, "model_id": model_id, "started_at": time.perf_counter(),
        "trace_id": trace.trace_id if trace else None,
        "seq": trace.seq if trace else None,
        "translation_epoch": trace.epoch if trace else None,
        "process_id": os.getpid(), "successful": None,
    }
    try:
        yield event
    except BaseException as exc:
        event["successful"] = False
        event["exception_type"] = type(exc).__name__
        raise
    finally:
        event["finished_at"] = time.perf_counter()
        event["elapsed_seconds"] = max(0.0, event["finished_at"] - event["started_at"])
        if trace is not None:
            trace.events.append(dict(event))
        save_execution_event(event)


def delivery_evidence(frame, *, retained_count: int = 0) -> dict:
    """Describe the route using observed work, not cached flags on old items."""
    trace = getattr(frame, "_execution_evidence", None)
    accepted = [item for item in frame.items if item.quality_status == "accepted"]
    cached = sum(bool(item.from_cache) for item in accepted)
    observed = bool(trace and trace.get("completed") and not trace.get("coalesced_unknown"))
    counts = trace.get("counts", {}) if observed else {}
    inferences = counts.get("local_inference", 0) + counts.get("api_inference", 0) if observed else None
    trusted = frame.diagnostics.get("trusted_cache_reuse", 0)
    if retained_count:
        route = "retained_with_inference" if inferences else "retained_confirmed" if observed else "retained_unobserved"
    elif accepted and cached == len(accepted) and trusted >= cached and inferences == 0:
        route = "trusted_cache_restore"
    elif inferences:
        route = "model_with_cache" if cached else "model"
    elif not accepted:
        route = "no_accepted_translation"
    else:
        route = "unobserved"
    return {
        "protocol_version": PROTOCOL,
        "delivery_route": route,
        "confirmed_retained_items": retained_count,
        "accepted_items": len(accepted), "cached_item_flags": cached,
        "trusted_cache_reuse": trusted,
        "model_calls_observed": observed,
        "inference_calls": inferences,
        "zero_inference_calls_proven": inferences == 0 if observed else None,
        "work": trace,
    }
