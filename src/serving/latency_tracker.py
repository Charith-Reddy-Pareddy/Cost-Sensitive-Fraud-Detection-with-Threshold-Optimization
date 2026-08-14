"""In-memory per-prediction latency tracking for the inference service."""

import threading

import numpy as np


class LatencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples_ms: list[float] = []

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._samples_ms.append(latency_ms)

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self._samples_ms)

    def percentiles(self) -> dict:
        samples = self.snapshot()
        if not samples:
            return {"count": 0, "p50_ms": None, "p95_ms": None}
        arr = np.array(samples)
        return {
            "count": len(arr),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }
