"""
User management module with clear, testable code.

This module contains exactly 4 entities:
1. User class
2. get_user_by_id function
3. create_user function
4. validate_email function

Expected relationships:
- create_user CALLS validate_email
- get_user_by_id CALLS User.__init__ (implicit)
- create_user CALLS User.__init__ (implicit)
"""

import re
from typing import Optional


class User:
    """User model with basic validation."""

    def __init__(self, user_id: int, email: str, name: str):
        """Initialize a new user."""
        self.user_id = user_id
        self.email = email
        self.name = name
        self.is_active = True

    def deactivate(self):
        """Deactivate the user account."""
        self.is_active = False
        return True


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def get_user_by_id(user_id: int) -> Optional[User]:
    """Retrieve user by ID from mock database."""
    # Mock data - in real app this would be a database call
    mock_users = {
        1: ("john@example.com", "John Doe"),
        2: ("jane@example.com", "Jane Smith")
    }

    if user_id in mock_users:
        email, name = mock_users[user_id]
        return User(user_id, email, name)
    return None


def create_user(email: str, name: str) -> User:
    """Create a new user with validation."""
    # Validate email format
    if not validate_email(email):
        raise ValueError("Invalid email format")

    # Generate mock ID
    user_id = len(name) + len(email)  # Simple mock ID generation

    # Create and return user
    return User(user_id, email, name)