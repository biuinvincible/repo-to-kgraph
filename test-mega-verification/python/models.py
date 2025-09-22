"""
Advanced Python test file with complex patterns.

This file contains 15 entities with intricate relationships:
- Abstract base classes with inheritance
- Decorators and property methods
- Async/await patterns
- Context managers
- Exception handling
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager


@dataclass
class UserProfile:
    """User profile data class with validation."""
    user_id: int
    email: str
    name: str
    is_active: bool = True

    def __post_init__(self):
        """Validate profile data after initialization."""
        if not self.email or '@' not in self.email:
            raise ValueError("Invalid email format")


class DatabaseError(Exception):
    """Custom database exception with error codes."""

    def __init__(self, message: str, error_code: int = 500):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = asyncio.get_event_loop().time()


class BaseRepository(ABC):
    """Abstract base class for repository pattern."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Abstract method for database connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Abstract method for cleanup."""
        pass

    @property
    def is_connected(self) -> bool:
        """Property to check connection status."""
        return self._connected

    @asynccontextmanager
    async def transaction(self):
        """Async context manager for database transactions."""
        try:
            await self.begin_transaction()
            yield self
            await self.commit()
        except Exception as e:
            await self.rollback()
            raise DatabaseError(f"Transaction failed: {e}", 503)

    async def begin_transaction(self):
        """Start database transaction."""
        if not self.is_connected:
            raise DatabaseError("Not connected to database", 500)

    async def commit(self):
        """Commit database transaction."""
        pass

    async def rollback(self):
        """Rollback database transaction."""
        pass


def retry_on_failure(max_retries: int = 3):
    """Decorator for automatic retry logic."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except DatabaseError as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator


class UserRepository(BaseRepository):
    """Concrete implementation of user repository."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._cache: Dict[int, UserProfile] = {}

    async def connect(self) -> bool:
        """Implement database connection."""
        try:
            # Simulate connection logic
            await asyncio.sleep(0.1)
            self._connected = True
            return True
        except Exception as e:
            raise DatabaseError(f"Connection failed: {e}", 502)

    async def disconnect(self) -> None:
        """Cleanup database connection."""
        self._connected = False
        self._cache.clear()

    @retry_on_failure(max_retries=3)
    async def get_user(self, user_id: int) -> Optional[UserProfile]:
        """Retrieve user by ID with caching and retry."""
        if user_id in self._cache:
            return self._cache[user_id]

        async with self.transaction():
            # Simulate database query
            if user_id > 0:
                profile = UserProfile(
                    user_id=user_id,
                    email=f"user{user_id}@example.com",
                    name=f"User {user_id}"
                )
                self._cache[user_id] = profile
                return profile

        return None

    @retry_on_failure(max_retries=5)
    async def save_user(self, profile: UserProfile) -> bool:
        """Save user profile with validation and retry."""
        try:
            # Validate using the profile's built-in validation
            profile.__post_init__()

            async with self.transaction():
                self._cache[profile.user_id] = profile
                return True

        except ValueError as e:
            raise DatabaseError(f"Validation error: {e}", 400)

    async def get_all_active_users(self) -> List[UserProfile]:
        """Get all active users with filtering."""
        return [
            profile for profile in self._cache.values()
            if profile.is_active
        ]


# Module-level factory function
async def create_user_repository(connection_string: str) -> UserRepository:
    """Factory function to create and initialize user repository."""
    repo = UserRepository(connection_string)
    await repo.connect()
    return repo


# Global configuration
DEFAULT_CONNECTION = "postgresql://localhost:5432/testdb"