import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import structlog
from prometheus_client import Counter, Gauge
import jwt
import hashlib
import aiofiles
import asyncio
from dataclasses import dataclass
from collections import defaultdict

logger = structlog.get_logger(__name__)

@dataclass
class RateLimitConfig:
    window_seconds: int
    max_requests: int

class SecurityManager:
    def __init__(
        self,
        jwt_secret: str,
        rate_limits: Dict[str, RateLimitConfig] = None,
        audit_log_path: str = "audit.log"
    ):
        self.jwt_secret = jwt_secret
        self.rate_limits = rate_limits or {
            "default": RateLimitConfig(60, 100),  # 100 requests per minute
            "metrics": RateLimitConfig(60, 300),  # 300 requests per minute
            "admin": RateLimitConfig(60, 50)      # 50 requests per minute
        }
        self.audit_log_path = audit_log_path
        
        # Rate limiting state
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.last_reset = defaultdict(float)
        
        # Security metrics
        self.auth_failures = Counter(
            'auth_failures_total',
            'Number of authentication failures',
            ['endpoint']
        )
        
        self.rate_limit_hits = Counter(
            'rate_limit_hits_total',
            'Number of rate limit violations',
            ['endpoint']
        )
        
        self.active_tokens = Gauge(
            'active_tokens_total',
            'Number of active API tokens'
        )

    async def authenticate_request(
        self,
        token: str,
        required_role: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """Authenticate a request using JWT"""
        try:
            # Verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            # Check expiration
            if payload.get('exp', 0) < time.time():
                await self.audit_log(
                    "auth_failure",
                    "Token expired",
                    {"token_id": payload.get('jti')}
                )
                return False, None
            
            # Check role if required
            if required_role and required_role not in payload.get('roles', []):
                await self.audit_log(
                    "auth_failure",
                    "Insufficient permissions",
                    {"token_id": payload.get('jti'), "required_role": required_role}
                )
                return False, None
            
            return True, payload
            
        except jwt.InvalidTokenError as e:
            await self.audit_log(
                "auth_failure",
                str(e),
                {"token": hashlib.sha256(token.encode()).hexdigest()}
            )
            return False, None

    async def check_rate_limit(
        self,
        endpoint: str,
        client_id: str
    ) -> bool:
        """Check if request is within rate limits"""
        limit_config = self.rate_limits.get(endpoint, self.rate_limits["default"])
        now = time.time()
        
        # Reset counter if window expired
        if now - self.last_reset[endpoint] > limit_config.window_seconds:
            self.request_counts[endpoint].clear()
            self.last_reset[endpoint] = now
        
        # Check and update counter
        current_count = self.request_counts[endpoint][client_id]
        if current_count >= limit_config.max_requests:
            self.rate_limit_hits.labels(endpoint=endpoint).inc()
            await self.audit_log(
                "rate_limit",
                "Rate limit exceeded",
                {
                    "endpoint": endpoint,
                    "client_id": client_id,
                    "count": current_count
                }
            )
            return False
        
        self.request_counts[endpoint][client_id] += 1
        return True

    async def audit_log(
        self,
        event_type: str,
        message: str,
        details: Optional[Dict] = None
    ):
        """Log security events to audit log"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "message": message,
                "details": details or {}
            }
            
            async with aiofiles.open(self.audit_log_path, 'a') as f:
                await f.write(f"{str(log_entry)}\n")
                
        except Exception as e:
            logger.error("Error writing to audit log",
                        error=str(e))

    def create_token(
        self,
        client_id: str,
        roles: List[str],
        expiry_hours: int = 24
    ) -> str:
        """Create a new JWT token"""
        payload = {
            'sub': client_id,
            'roles': roles,
            'iat': int(time.time()),
            'exp': int(time.time() + expiry_hours * 3600),
            'jti': hashlib.sha256(f"{client_id}{time.time()}".encode()).hexdigest()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        self.active_tokens.inc()
        
        return token

    def revoke_token(self, token: str):
        """Revoke a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            # In a production system, you would add the token to a blacklist
            # For this example, we just decrement the active tokens counter
            self.active_tokens.dec()
            
        except jwt.InvalidTokenError:
            pass

    async def start_cleanup_task(self):
        """Start periodic cleanup of rate limit counters"""
        while True:
            now = time.time()
            for endpoint in list(self.request_counts.keys()):
                if now - self.last_reset[endpoint] > self.rate_limits[endpoint].window_seconds:
                    self.request_counts[endpoint].clear()
                    self.last_reset[endpoint] = now
            await asyncio.sleep(60)  # Run every minute 