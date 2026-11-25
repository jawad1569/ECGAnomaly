"""Minimal monitoring: request count and latency, in memory.

- A middleware times every request, adds it to the counters and writes one log line:
  `request method=POST path=/predict status=200 latency_ms=812.4`
- GET /stats (alias /metrics) returns the counters as JSON.

Counters live in the process, so they reset when the container restarts.
Calls to /stats and /metrics are not counted, so looking at them doesn't change them.
"""

import logging
import sys
import threading
import time

from fastapi import APIRouter, Request

from src.utils.model_loader import get_model_version

NOT_COUNTED = {"/stats", "/metrics"}

logger = logging.getLogger("ecg_api")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class RequestStats:
    """Thread-safe request counters, total and per endpoint."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.started_at = time.time()
            self.count = 0
            self.total_ms = 0.0
            self.client_errors = 0  # 4xx
            self.server_errors = 0  # 5xx
            self.per_endpoint = {}

    def record(self, method: str, path: str, status: int, latency_ms: float):
        key = f"{method} {path}"
        with self._lock:
            self.count += 1
            self.total_ms += latency_ms
            if 400 <= status < 500:
                self.client_errors += 1
            elif status >= 500:
                self.server_errors += 1
            ep = self.per_endpoint.setdefault(key, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            ep["count"] += 1
            ep["total_ms"] += latency_ms
            ep["max_ms"] = max(ep["max_ms"], latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "request_count": self.count,
                "avg_latency_ms": round(self.total_ms / self.count, 2) if self.count else 0.0,
                "client_errors_4xx": self.client_errors,
                "server_errors_5xx": self.server_errors,
                "uptime_s": round(time.time() - self.started_at, 1),
                "model_version": get_model_version(),
                "per_endpoint": {
                    key: {
                        "count": ep["count"],
                        "avg_latency_ms": round(ep["total_ms"] / ep["count"], 2),
                        "max_latency_ms": round(ep["max_ms"], 2),
                    }
                    for key, ep in sorted(self.per_endpoint.items())
                },
            }


stats = RequestStats()
router = APIRouter()


async def track_requests(request: Request, call_next):
    """HTTP middleware: time the request, count it, log it."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        if path not in NOT_COUNTED:
            latency_ms = (time.perf_counter() - start) * 1000
            stats.record(request.method, path, status, latency_ms)
            logger.info(
                "request method=%s path=%s status=%d latency_ms=%.1f",
                request.method,
                path,
                status,
                latency_ms,
            )


@router.get("/stats", tags=["System"])
def get_stats():
    """Request count and average latency since the service started (total and per endpoint)."""
    return stats.snapshot()


@router.get("/metrics", tags=["System"], include_in_schema=False)
def get_metrics():
    """Alias of /stats."""
    return stats.snapshot()
