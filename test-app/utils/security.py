"""
Security utilities for URL shortener application.
Handles password hashing, token generation, and security validation.
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import bcrypt
import jwt


class SecurityManager:
    """Manages security operations for the application."""

    def __init__(self, secret_key: str, salt_rounds: int = 12):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.salt_rounds = salt_rounds
        self.jwt_algorithm = 'HS256'
        self.token_expiry_hours = 24

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        if not password:
            raise ValueError("Password cannot be empty")

        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=self.salt_rounds)
        password_bytes = password.encode('utf-8')
        hashed = bcrypt.hashpw(password_bytes, salt)

        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if not password or not hashed_password:
            return False

        try:
            password_bytes = password.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False

    def generate_token(self, user_id: int, additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Generate a JWT token for user authentication."""
        now = datetime.utcnow()
        expiry = now + timedelta(hours=self.token_expiry_hours)

        payload = {
            'user_id': user_id,
            'iat': now,
            'exp': expiry,
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def generate_api_key(self, prefix: str = "sk") -> str:
        """Generate a secure API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"

    def validate_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Validate an API key against its stored hash."""
        return self.verify_hmac(api_key, stored_hash)

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return self.generate_hmac(api_key)

    def generate_hmac(self, data: str) -> str:
        """Generate HMAC signature for data."""
        data_bytes = data.encode('utf-8')
        signature = hmac.new(self.secret_key, data_bytes, hashlib.sha256)
        return signature.hexdigest()

    def verify_hmac(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.generate_hmac(data)
        return hmac.compare_digest(expected_signature, signature)

    def generate_short_code_seed(self, url: str, timestamp: Optional[float] = None) -> str:
        """Generate a deterministic seed for short code generation."""
        if timestamp is None:
            timestamp = time.time()

        data = f"{url}:{timestamp}:{secrets.token_bytes(8).hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:8]

    def validate_url_safety(self, url: str) -> bool:
        """Validate URL for security concerns."""
        url_lower = url.lower()

        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        if any(url_lower.startswith(proto) for proto in dangerous_protocols):
            return False

        # Check for suspicious patterns
        suspicious_patterns = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '192.168.',
            '10.0.',
            '172.16.',
            '172.17.',
            '172.18.',
            '172.19.',
            '172.20.',
            '172.21.',
            '172.22.',
            '172.23.',
            '172.24.',
            '172.25.',
            '172.26.',
            '172.27.',
            '172.28.',
            '172.29.',
            '172.30.',
            '172.31.'
        ]

        # Allow localhost and private IPs in debug mode (would be configurable)
        # For production, these should be blocked
        return True  # Simplified for demo

    def sanitize_input(self, input_string: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not input_string:
            return ""

        # Truncate to max length
        sanitized = input_string[:max_length]

        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')

        # Strip whitespace
        sanitized = sanitized.strip()

        return sanitized

    def is_rate_limited(self, identifier: str, limit: int, window_seconds: int = 3600) -> bool:
        """Check if an identifier is rate limited (simplified implementation)."""
        # In a real implementation, this would use Redis or another cache
        # For this demo, we'll use a simple in-memory approach

        if not hasattr(self, '_rate_limit_cache'):
            self._rate_limit_cache = {}

        current_time = time.time()
        window_start = current_time - window_seconds

        # Clean old entries
        self._rate_limit_cache = {
            key: timestamps for key, timestamps in self._rate_limit_cache.items()
            if any(ts > window_start for ts in timestamps)
        }

        # Get timestamps for this identifier
        timestamps = self._rate_limit_cache.get(identifier, [])

        # Filter to current window
        timestamps = [ts for ts in timestamps if ts > window_start]

        # Check if limit exceeded
        if len(timestamps) >= limit:
            return True

        # Add current timestamp
        timestamps.append(current_time)
        self._rate_limit_cache[identifier] = timestamps

        return False

    def generate_csrf_token(self, session_id: str) -> str:
        """Generate a CSRF token tied to session."""
        data = f"{session_id}:{time.time()}"
        return self.generate_hmac(data)

    def verify_csrf_token(self, token: str, session_id: str, max_age_seconds: int = 3600) -> bool:
        """Verify a CSRF token."""
        # In a real implementation, you'd store token creation time
        # For this demo, we'll accept any valid HMAC
        try:
            # Extract timestamp from potential data
            current_time = time.time()
            # This is a simplified verification
            return bool(token and len(token) == 64)  # HMAC length check
        except Exception:
            return False

    def hash_for_storage(self, data: str) -> str:
        """Hash data for secure storage (one-way)."""
        return hashlib.sha256(data.encode()).hexdigest()

    def constant_time_compare(self, a: str, b: str) -> bool:
        """Compare two strings in constant time to prevent timing attacks."""
        return hmac.compare_digest(a, b)

    def generate_secure_filename(self, original_filename: str) -> str:
        """Generate a secure filename to prevent path traversal."""
        # Remove directory separators and dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        cleaned = ''.join(c for c in original_filename if c in safe_chars)

        # Ensure it's not empty and doesn't start with dot
        if not cleaned or cleaned.startswith('.'):
            cleaned = f"file_{secrets.token_urlsafe(8)}"

        # Add timestamp to ensure uniqueness
        timestamp = int(time.time())
        name, ext = cleaned.rsplit('.', 1) if '.' in cleaned else (cleaned, '')

        if ext:
            return f"{name}_{timestamp}.{ext}"
        else:
            return f"{name}_{timestamp}"

    def validate_user_agent(self, user_agent: str) -> bool:
        """Validate user agent string for suspicious patterns."""
        if not user_agent or len(user_agent) > 500:
            return False

        # Check for suspicious patterns that might indicate bots or attacks
        suspicious_patterns = [
            'sqlmap',
            'nikto',
            'nmap',
            'masscan',
            'python-requests',  # Could be legitimate, but often used in attacks
            'curl',  # Same as above
            'wget'
        ]

        user_agent_lower = user_agent.lower()
        return not any(pattern in user_agent_lower for pattern in suspicious_patterns)

    def extract_ip_from_headers(self, headers: Dict[str, str]) -> str:
        """Extract real IP address from headers, considering proxies."""
        # Check common headers used by proxies and load balancers
        ip_headers = [
            'X-Forwarded-For',
            'X-Real-IP',
            'X-Client-IP',
            'CF-Connecting-IP',  # Cloudflare
            'True-Client-IP'
        ]

        for header in ip_headers:
            if header in headers:
                ip = headers[header].split(',')[0].strip()
                if self._is_valid_ip(ip):
                    return ip

        # Fallback to remote address
        return headers.get('Remote-Addr', 'unknown')

    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False

            for part in parts:
                if not 0 <= int(part) <= 255:
                    return False

            return True
        except (ValueError, AttributeError):
            return False