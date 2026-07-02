import time
from collections import defaultdict

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    # ponytail: single-process in-memory window; move to a shared store (e.g. Redis)
    # if the app ever runs as more than one instance behind a load balancer
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._windows: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))

    async def dispatch(self, request: Request, call_next):
        key = request.headers.get("x-api-key") or (
            # LAST hop of X-Forwarded-For: caddy is the one trusted proxy (the app
            # binds loopback-only) and APPENDS the real peer IP, so the rightmost
            # value is the one caddy observed. The leftmost is client-supplied and
            # spoofable — keying on it would let a caller rotate XFF to evade the limit.
            (request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or None)
            or (request.client.host if request.client else "unknown")
        )
        now = time.monotonic()
        if len(self._windows) > 10_000:
            # ponytail: single-process memory bound — drop windows whose 60s elapsed;
            # switch to redis with TTL if this ever runs multi-instance
            self._windows = defaultdict(
                lambda: (0, 0.0),
                {k: v for k, v in self._windows.items() if now - v[1] < 60},
            )
        count, window_start = self._windows[key]
        if now - window_start >= 60:
            count, window_start = 0, now
        count += 1
        self._windows[key] = (count, window_start)
        if count > self.requests_per_minute:
            retry_after = max(0, 60 - (now - window_start))
            return JSONResponse(
                {"error": "rate limit exceeded"}, status_code=429,
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.requests_per_minute - count))
        return response


app = FastAPI(title="uaisearch")
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
