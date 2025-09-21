"""
User service for managing user accounts and authentication.
Handles user creation, authentication, and authorization.
"""

import hashlib
from datetime import datetime
from typing import List, Optional

from models.user import User, UserRole
from utils.database import Database
from utils.security import SecurityManager


class UserService:
    """Service for managing user operations."""

    def __init__(self, database: Database, security: SecurityManager):
        self.database = database
        self.security = security

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user account."""

        # Validate input
        await self._validate_user_input(username, email)

        # Hash password
        password_hash = self.security.hash_password(password)

        # Insert user into database
        query = """
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """
        params = (username, email, password_hash, role.value)

        user_id = await self.database.execute_query(query, params, return_id=True)

        # Return user object
        return User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            created_at=datetime.utcnow(),
            last_login=None,
            is_active=True
        )

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""

        user = await self.get_user_by_username(username)
        if not user:
            return None

        if not user.is_active:
            return None

        # Verify password
        if not self.security.verify_password(password, user.password_hash):
            return None

        # Update last login
        await self._update_last_login(user.id)
        user.update_last_login()

        return user

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        query = """
            SELECT id, username, email, password_hash, role, created_at, last_login, is_active
            FROM users WHERE id = ?
        """

        result = await self.database.fetch_one(query, (user_id,))

        if not result:
            return None

        return self._create_user_from_row(result)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        query = """
            SELECT id, username, email, password_hash, role, created_at, last_login, is_active
            FROM users WHERE username = ?
        """

        result = await self.database.fetch_one(query, (username,))

        if not result:
            return None

        return self._create_user_from_row(result)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        query = """
            SELECT id, username, email, password_hash, role, created_at, last_login, is_active
            FROM users WHERE email = ?
        """

        result = await self.database.fetch_one(query, (email,))

        if not result:
            return None

        return self._create_user_from_row(result)

    async def get_all_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get all users with pagination."""
        query = """
            SELECT id, username, email, password_hash, role, created_at, last_login, is_active
            FROM users
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """

        results = await self.database.fetch_all(query, (limit, offset))

        return [self._create_user_from_row(row) for row in results]

    async def update_user_role(self, user_id: int, new_role: UserRole) -> bool:
        """Update user's role."""
        query = "UPDATE users SET role = ? WHERE id = ?"

        rows_affected = await self.database.execute_query(query, (new_role.value, user_id))
        return rows_affected > 0

    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account."""
        query = "UPDATE users SET is_active = FALSE WHERE id = ?"

        rows_affected = await self.database.execute_query(query, (user_id,))
        return rows_affected > 0

    async def activate_user(self, user_id: int) -> bool:
        """Activate a user account."""
        query = "UPDATE users SET is_active = TRUE WHERE id = ?"

        rows_affected = await self.database.execute_query(query, (user_id,))
        return rows_affected > 0

    async def change_password(self, user_id: int, new_password: str) -> bool:
        """Change user's password."""
        password_hash = self.security.hash_password(new_password)

        query = "UPDATE users SET password_hash = ? WHERE id = ?"

        rows_affected = await self.database.execute_query(query, (password_hash, user_id))
        return rows_affected > 0

    async def update_user_profile(
        self,
        user_id: int,
        email: Optional[str] = None,
        username: Optional[str] = None
    ) -> bool:
        """Update user profile information."""
        updates = []
        params = []

        if email is not None:
            # Check if email is already taken
            existing_user = await self.get_user_by_email(email)
            if existing_user and existing_user.id != user_id:
                raise ValueError("Email is already in use")

            updates.append("email = ?")
            params.append(email)

        if username is not None:
            # Check if username is already taken
            existing_user = await self.get_user_by_username(username)
            if existing_user and existing_user.id != user_id:
                raise ValueError("Username is already in use")

            updates.append("username = ?")
            params.append(username)

        if not updates:
            return False

        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"

        rows_affected = await self.database.execute_query(query, params)
        return rows_affected > 0

    async def _validate_user_input(self, username: str, email: str) -> None:
        """Validate user input for creation."""

        # Check username requirements
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters long")

        if len(username) > 50:
            raise ValueError("Username must be less than 50 characters")

        # Check if username already exists
        existing_user = await self.get_user_by_username(username)
        if existing_user:
            raise ValueError("Username is already taken")

        # Check email format (basic validation)
        if not email or '@' not in email:
            raise ValueError("Invalid email format")

        # Check if email already exists
        existing_user = await self.get_user_by_email(email)
        if existing_user:
            raise ValueError("Email is already registered")

    async def _update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp."""
        query = "UPDATE users SET last_login = ? WHERE id = ?"
        await self.database.execute_query(query, (datetime.utcnow(), user_id))

    def _create_user_from_row(self, row: dict) -> User:
        """Create User object from database row."""
        return User(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            password_hash=row['password_hash'],
            role=UserRole(row['role']),
            created_at=row['created_at'],
            last_login=row['last_login'],
            is_active=bool(row['is_active'])
        )

    async def get_user_stats(self, user_id: int) -> dict:
        """Get statistics for a user."""
        query = """
            SELECT COUNT(*) as total_urls,
                   SUM(click_count) as total_clicks,
                   AVG(click_count) as avg_clicks_per_url
            FROM urls WHERE user_id = ? AND is_active = TRUE
        """

        result = await self.database.fetch_one(query, (user_id,))

        return {
            'total_urls': result['total_urls'] or 0,
            'total_clicks': result['total_clicks'] or 0,
            'avg_clicks_per_url': result['avg_clicks_per_url'] or 0.0
        }