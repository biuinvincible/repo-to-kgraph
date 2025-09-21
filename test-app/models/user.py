"""
User model for URL shortener service.
Handles user authentication and authorization.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class UserRole(Enum):
    """User roles for authorization."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

    def has_permission(self, required_role: 'UserRole') -> bool:
        """Check if this role has permission for required role."""
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.MODERATOR: 2,
            UserRole.ADMIN: 3
        }
        return role_hierarchy[self] >= role_hierarchy[required_role]


@dataclass
class User:
    """Represents a user in the system."""

    id: int
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

    def can_create_urls(self) -> bool:
        """Check if user can create URLs."""
        return self.is_active

    def can_manage_users(self) -> bool:
        """Check if user can manage other users."""
        return self.is_active and self.role in [UserRole.ADMIN, UserRole.MODERATOR]

    def can_view_analytics(self, url_user_id: Optional[int]) -> bool:
        """Check if user can view analytics for a URL."""
        if not self.is_active:
            return False

        # Admins can view all analytics
        if self.role == UserRole.ADMIN:
            return True

        # Users can only view their own URL analytics
        return url_user_id == self.id

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary representation."""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }

        if include_sensitive:
            data['password_hash'] = self.password_hash

        return data

    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()

    def is_admin(self) -> bool:
        """Check if user is an administrator."""
        return self.role == UserRole.ADMIN

    def is_moderator_or_higher(self) -> bool:
        """Check if user is moderator or admin."""
        return self.role in [UserRole.MODERATOR, UserRole.ADMIN]