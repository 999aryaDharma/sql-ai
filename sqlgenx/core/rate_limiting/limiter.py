"""Rate limiting and circuit breaker for API calls."""
import time
import hashlib
from collections import deque
from typing import Dict, Tuple, Optional
from threading import Lock


class RateLimiter:
    """
    Token bucket rate limiter to prevent API quota exhaustion.
    Enforces maximum requests per time window.
    """
    
    def __init__(
        self,
        max_requests: int = 10,
        time_window: int = 60,
        burst_allowance: int = 2
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed per time window
            time_window: Time window in seconds (default: 60s)
            burst_allowance: Extra requests allowed for bursts
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_allowance = burst_allowance
        self.requests = deque()  # Timestamps of recent requests
        self.lock = Lock()
        
        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.rate_limit_hits = 0
    
    def can_proceed(self) -> bool:
        """
        Check if we can make another API call without waiting.
        
        Returns:
            True if request can proceed immediately
        """
        with self.lock:
            now = time.time()
            
            # Remove requests outside the time window
            while self.requests and now - self.requests[0] >= self.time_window:
                self.requests.popleft()
            
            # Check if under limit (including burst allowance)
            return len(self.requests) < (self.max_requests + self.burst_allowance)
    
    def wait_if_needed(self) -> float:
        """
        Wait if rate limit would be exceeded.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and now - self.requests[0] >= self.time_window:
                self.requests.popleft()
            
            # Check if we need to wait
            if len(self.requests) >= self.max_requests:
                # Calculate wait time until oldest request expires
                oldest_request = self.requests[0]
                wait_time = self.time_window - (now - oldest_request) + 0.1  # Add 100ms buffer
                
                if wait_time > 0:
                    self.rate_limit_hits += 1
                    self.total_wait_time += wait_time
                    
                    time.sleep(wait_time)
                    
                    # Clean up after sleeping
                    now = time.time()
                    while self.requests and now - self.requests[0] >= self.time_window:
                        self.requests.popleft()
                    
                    # Record this request
                    self.requests.append(now)
                    self.total_requests += 1
                    
                    return wait_time
            
            # No wait needed, record request
            self.requests.append(now)
            self.total_requests += 1
            
            return 0.0
    
    def record_request(self) -> None:
        """Record an API request without waiting (for async usage)."""
        with self.lock:
            now = time.time()
            self.requests.append(now)
            self.total_requests += 1
    
    def get_current_usage(self) -> Dict[str, any]:
        """
        Get current rate limiter usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and now - self.requests[0] >= self.time_window:
                self.requests.popleft()
            
            current_requests = len(self.requests)
            remaining_requests = max(0, self.max_requests - current_requests)
            
            return {
                'current_requests': current_requests,
                'max_requests': self.max_requests,
                'remaining_requests': remaining_requests,
                'time_window': self.time_window,
                'total_requests': self.total_requests,
                'rate_limit_hits': self.rate_limit_hits,
                'total_wait_time': self.total_wait_time,
            }
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        with self.lock:
            self.requests.clear()
            self.total_requests = 0
            self.total_wait_time = 0.0
            self.rate_limit_hits = 0


class CircuitBreaker:
    """
    Circuit breaker to prevent infinite loops in query refinement.
    Tracks attempts and enforces cooldown periods.
    """
    
    # Circuit states
    STATE_CLOSED = "closed"    # Normal operation
    STATE_OPEN = "open"        # Too many failures, reject requests
    STATE_HALF_OPEN = "half_open"  # Testing if system recovered
    
    def __init__(
        self,
        max_attempts: int = 1,
        cooldown_seconds: int = 300,
        failure_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            max_attempts: Maximum refinement attempts per query
            cooldown_seconds: Cooldown period before reset (default: 5 minutes)
            failure_threshold: Failures before opening circuit
        """
        self.max_attempts = max_attempts
        self.cooldown = cooldown_seconds
        self.failure_threshold = failure_threshold
        
        # Track attempts per query hash
        self.attempts: Dict[str, Tuple[int, float]] = {}  # query_hash -> (count, last_attempt_time)
        
        # Track error patterns
        self.error_patterns: Dict[str, int] = {}  # error_hash -> count
        
        # Circuit state
        self.state = self.STATE_CLOSED
        self.state_change_time = time.time()
        
        self.lock = Lock()
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        return hashlib.sha256(query.encode()).hexdigest()[:12]
    
    def _hash_error(self, error: str) -> str:
        """Generate hash for error message."""
        return hashlib.sha256(error.encode()).hexdigest()[:8]
    
    def can_attempt(self, query: str) -> bool:
        """
        Check if we can attempt query generation/refinement.
        
        Args:
            query: User query string
            
        Returns:
            True if attempt is allowed
        """
        with self.lock:
            query_hash = self._hash_query(query)
            now = time.time()
            
            # Check circuit state
            if self.state == self.STATE_OPEN:
                # Check if cooldown period has passed
                if now - self.state_change_time >= self.cooldown:
                    self.state = self.STATE_HALF_OPEN
                    self.state_change_time = now
                else:
                    return False
            
            # Check query-specific attempts
            if query_hash not in self.attempts:
                return True
            
            count, last_time = self.attempts[query_hash]
            
            # Reset if cooldown period has passed
            if now - last_time >= self.cooldown:
                del self.attempts[query_hash]
                return True
            
            # Check attempt limit
            return count < self.max_attempts
    
    def record_attempt(self, query: str, error: Optional[str] = None) -> None:
        """
        Record a refinement attempt.
        
        Args:
            query: User query string
            error: Optional error message
        """
        with self.lock:
            query_hash = self._hash_query(query)
            now = time.time()
            
            # Update attempt count
            if query_hash in self.attempts:
                count, _ = self.attempts[query_hash]
                self.attempts[query_hash] = (count + 1, now)
            else:
                self.attempts[query_hash] = (1, now)
            
            # Track error pattern if provided
            if error:
                error_hash = self._hash_error(error)
                self.error_patterns[error_hash] = self.error_patterns.get(error_hash, 0) + 1
                
                # Check if we're seeing too many of the same error
                if self.error_patterns[error_hash] >= self.failure_threshold:
                    self._open_circuit()
    
    def record_success(self, query: str) -> None:
        """
        Record a successful attempt.
        
        Args:
            query: User query string
        """
        with self.lock:
            query_hash = self._hash_query(query)
            
            # Reset attempts for this query
            if query_hash in self.attempts:
                del self.attempts[query_hash]
            
            # Close circuit if in half-open state
            if self.state == self.STATE_HALF_OPEN:
                self._close_circuit()
    
    def _open_circuit(self) -> None:
        """Open circuit (block requests)."""
        self.state = self.STATE_OPEN
        self.state_change_time = time.time()
    
    def _close_circuit(self) -> None:
        """Close circuit (allow requests)."""
        self.state = self.STATE_CLOSED
        self.state_change_time = time.time()
        self.error_patterns.clear()
    
    def get_remaining_attempts(self, query: str) -> int:
        """
        Get remaining attempts for a query.
        
        Args:
            query: User query string
            
        Returns:
            Number of remaining attempts
        """
        with self.lock:
            query_hash = self._hash_query(query)
            now = time.time()
            
            if query_hash not in self.attempts:
                return self.max_attempts
            
            count, last_time = self.attempts[query_hash]
            
            # Reset if cooldown passed
            if now - last_time >= self.cooldown:
                return self.max_attempts
            
            return max(0, self.max_attempts - count)
    
    def is_error_repeating(self, error: str, threshold: int = 2) -> bool:
        """
        Check if same error is repeating.
        
        Args:
            error: Error message
            threshold: Repetition threshold
            
        Returns:
            True if error has been seen threshold times
        """
        with self.lock:
            error_hash = self._hash_error(error)
            return self.error_patterns.get(error_hash, 0) >= threshold
    
    def reset(self, query: Optional[str] = None) -> None:
        """
        Reset circuit breaker.
        
        Args:
            query: Optional specific query to reset (resets all if None)
        """
        with self.lock:
            if query:
                query_hash = self._hash_query(query)
                if query_hash in self.attempts:
                    del self.attempts[query_hash]
            else:
                self.attempts.clear()
                self.error_patterns.clear()
                self._close_circuit()
    
    def get_stats(self) -> Dict[str, any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                'state': self.state,
                'active_queries': len(self.attempts),
                'error_patterns': len(self.error_patterns),
                'cooldown_seconds': self.cooldown,
                'max_attempts': self.max_attempts,
            }


# Global instances (can be overridden per workspace)
global_rate_limiter = RateLimiter(max_requests=10, time_window=60)
global_circuit_breaker = CircuitBreaker(max_attempts=1, cooldown_seconds=300)