from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60) -> None:
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> tuple[bool, str]:
        now = time.time()
        self.requests[ip] = [timestamp for timestamp in self.requests[ip] if timestamp > now - 60]
        if len(self.requests[ip]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute}/min"
        self.requests[ip].append(now)
        return True, "OK"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        allowed, message = self.limiter.is_allowed(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Too Many Requests", "detail": message},
            )
        return await call_next(request)
