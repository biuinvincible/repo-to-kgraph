"""
Service layer with complex dependencies and patterns.

Cross-file imports and sophisticated relationships:
- Multiple inheritance
- Mixin patterns
- Dependency injection
- Event handling
"""

import asyncio
from typing import List, Dict, Set, Callable, Any
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict

# Cross-file imports creating IMPORT relationships
from models import (
    UserProfile, UserRepository, DatabaseError,
    BaseRepository, create_user_repository, DEFAULT_CONNECTION
)


class EventType(Enum):
    """Event types for the observer pattern."""
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_LOGIN = "user_login"


class EventListener(ABC):
    """Abstract base class for event listeners."""

    @abstractmethod
    async def handle_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Handle a specific event."""
        pass


class CacheMixin:
    """Mixin providing caching functionality."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_from_cache(self, key: str) -> Any:
        """Retrieve item from cache."""
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        else:
            self._cache_misses += 1
            return None

    def add_to_cache(self, key: str, value: Any) -> None:
        """Add item to cache."""
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }


class MetricsMixin:
    """Mixin for tracking performance metrics."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)

    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric."""
        self._metrics[name].append(value)

    def get_average_metric(self, name: str) -> float:
        """Calculate average for a metric."""
        values = self._metrics.get(name, [])
        return sum(values) / len(values) if values else 0.0


class UserService(CacheMixin, MetricsMixin, EventListener):
    """
    User service with multiple inheritance and complex patterns.

    Inherits from multiple mixins and implements EventListener interface.
    """

    def __init__(self, repository: UserRepository):
        # Initialize parent classes
        CacheMixin.__init__(self)
        MetricsMixin.__init__(self)

        self.repository = repository
        self._listeners: Set[EventListener] = set()

    def add_event_listener(self, listener: EventListener) -> None:
        """Add an event listener."""
        self._listeners.add(listener)

    def remove_event_listener(self, listener: EventListener) -> None:
        """Remove an event listener."""
        self._listeners.discard(listener)

    async def _notify_listeners(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Notify all registered event listeners."""
        tasks = [
            listener.handle_event(event_type, data)
            for listener in self._listeners
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Handle events as a listener (implements EventListener)."""
        if event_type == EventType.USER_LOGIN:
            user_id = data.get('user_id')
            if user_id:
                self.record_metric('user_logins', 1.0)

    async def create_user(self, email: str, name: str) -> UserProfile:
        """Create a new user with event notification."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Create user profile
            profile = UserProfile(
                user_id=hash(email) % 10000,  # Simple ID generation
                email=email,
                name=name
            )

            # Save to repository
            success = await self.repository.save_user(profile)
            if not success:
                raise DatabaseError("Failed to save user", 500)

            # Cache the user
            cache_key = f"user:{profile.user_id}"
            self.add_to_cache(cache_key, profile)

            # Notify listeners
            await self._notify_listeners(EventType.USER_CREATED, {
                'user_id': profile.user_id,
                'email': profile.email,
                'name': profile.name
            })

            return profile

        finally:
            # Record performance metric
            duration = asyncio.get_event_loop().time() - start_time
            self.record_metric('create_user_duration', duration)

    async def get_user_by_id(self, user_id: int) -> UserProfile:
        """Get user by ID with caching."""
        cache_key = f"user:{user_id}"

        # Try cache first
        cached_user = self.get_from_cache(cache_key)
        if cached_user:
            return cached_user

        # Fetch from repository
        user = await self.repository.get_user(user_id)
        if user:
            self.add_to_cache(cache_key, user)

        return user

    async def update_user(self, user_id: int, **updates) -> UserProfile:
        """Update user information."""
        user = await self.get_user_by_id(user_id)
        if not user:
            raise DatabaseError(f"User {user_id} not found", 404)

        # Update fields
        for field, value in updates.items():
            if hasattr(user, field):
                setattr(user, field, value)

        # Save changes
        await self.repository.save_user(user)

        # Update cache
        cache_key = f"user:{user_id}"
        self.add_to_cache(cache_key, user)

        # Notify listeners
        await self._notify_listeners(EventType.USER_UPDATED, {
            'user_id': user_id,
            'updates': updates
        })

        return user

    async def get_user_statistics(self) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        active_users = await self.repository.get_all_active_users()

        return {
            'total_active_users': len(active_users),
            'cache_stats': self.cache_stats,
            'average_create_time': self.get_average_metric('create_user_duration'),
            'total_logins': sum(self._metrics.get('user_logins', []))
        }


class EmailNotificationListener(EventListener):
    """Email notification service implementing EventListener."""

    def __init__(self, smtp_server: str):
        self.smtp_server = smtp_server
        self._sent_emails = 0

    async def handle_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Send email notifications for user events."""
        if event_type == EventType.USER_CREATED:
            await self._send_welcome_email(data)
        elif event_type == EventType.USER_UPDATED:
            await self._send_update_notification(data)

    async def _send_welcome_email(self, user_data: Dict[str, Any]) -> None:
        """Send welcome email to new user."""
        # Simulate email sending
        await asyncio.sleep(0.05)
        self._sent_emails += 1

    async def _send_update_notification(self, user_data: Dict[str, Any]) -> None:
        """Send update notification email."""
        # Simulate email sending
        await asyncio.sleep(0.03)
        self._sent_emails += 1

    @property
    def emails_sent(self) -> int:
        """Get total number of emails sent."""
        return self._sent_emails


# Factory function with dependency injection
async def create_user_service_with_notifications(
    connection_string: str = DEFAULT_CONNECTION,
    smtp_server: str = "localhost:587"
) -> UserService:
    """
    Factory function creating fully configured user service.

    This function creates CALLS relationships to multiple other functions
    and demonstrates dependency injection pattern.
    """
    # Create repository (CALLS create_user_repository)
    repository = await create_user_repository(connection_string)

    # Create service (CALLS UserService.__init__)
    service = UserService(repository)

    # Create email listener (CALLS EmailNotificationListener.__init__)
    email_listener = EmailNotificationListener(smtp_server)

    # Wire up dependencies (CALLS service.add_event_listener)
    service.add_event_listener(email_listener)
    service.add_event_listener(service)  # Self-registration for metrics

    return service


# Global service instance
_global_service_instance: UserService = None


async def get_global_service() -> UserService:
    """Get or create global service instance (Singleton pattern)."""
    global _global_service_instance

    if _global_service_instance is None:
        _global_service_instance = await create_user_service_with_notifications()

    return _global_service_instance