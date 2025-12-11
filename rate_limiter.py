import time
import threading
from typing import Dict
import config

class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate = float(rate)
        self.capacity = int(burst)
        self._tokens = self.capacity
        self._last_refill_time = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill_time
            self._last_refill_time = now

            # Refill tokens
            self._tokens += elapsed * self.rate
            self._tokens = min(self.capacity, self._tokens)

            # Check if there are enough tokens and consume
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

class RateLimiterManager:
    def __init__(self, default_rate=config.DEFAULT_RATE_LIMIT, default_burst=config.DEFAULT_BURST_LIMIT):
        self.buckets: Dict[str, TokenBucket] = {}
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if key not in self.buckets:
            with self.lock:
                # Double-check in case another thread created it
                if key not in self.buckets:
                    self.buckets[key] = TokenBucket(self.default_rate, self.default_burst)
        
        return self.buckets[key].consume(1)

    def set_rate(self, key: str, rate: float, burst: int):
        with self.lock:
            self.buckets[key] = TokenBucket(rate, burst)
