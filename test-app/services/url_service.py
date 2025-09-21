"""
URL service for managing shortened URLs.
Handles creation, retrieval, and management of URLs.
"""

import asyncio
import hashlib
import random
import string
from datetime import datetime
from typing import List, Optional

from models.url import URL
from utils.database import Database
from utils.security import SecurityManager


class UrlService:
    """Service for managing URL operations."""

    def __init__(self, database: Database, security: SecurityManager):
        self.database = database
        self.security = security
        self.default_code_length = 6

    async def create_url(
        self,
        original_url: str,
        user_id: Optional[int] = None,
        custom_code: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> URL:
        """Create a new shortened URL."""

        # Generate or validate short code
        if custom_code:
            short_code = await self._validate_custom_code(custom_code)
        else:
            short_code = await self._generate_unique_code()

        # Insert URL into database
        query = """
            INSERT INTO urls (short_code, original_url, user_id, title, description, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (short_code, original_url, user_id, title, description, expires_at)

        url_id = await self.database.execute_query(query, params, return_id=True)

        # Return URL object
        return URL(
            id=url_id,
            short_code=short_code,
            original_url=original_url,
            user_id=user_id,
            title=title,
            description=description,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_active=True,
            click_count=0
        )

    async def get_url_by_code(self, short_code: str) -> Optional[URL]:
        """Retrieve URL by short code."""
        query = """
            SELECT id, short_code, original_url, user_id, title, description,
                   created_at, expires_at, is_active, click_count
            FROM urls WHERE short_code = ?
        """

        result = await self.database.fetch_one(query, (short_code,))

        if not result:
            return None

        return URL(
            id=result['id'],
            short_code=result['short_code'],
            original_url=result['original_url'],
            user_id=result['user_id'],
            title=result['title'],
            description=result['description'],
            created_at=result['created_at'],
            expires_at=result['expires_at'],
            is_active=bool(result['is_active']),
            click_count=result['click_count']
        )

    async def get_urls_by_user(self, user_id: int, limit: int = 50, offset: int = 0) -> List[URL]:
        """Get all URLs created by a user."""
        query = """
            SELECT id, short_code, original_url, user_id, title, description,
                   created_at, expires_at, is_active, click_count
            FROM urls WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """

        results = await self.database.fetch_all(query, (user_id, limit, offset))

        return [
            URL(
                id=row['id'],
                short_code=row['short_code'],
                original_url=row['original_url'],
                user_id=row['user_id'],
                title=row['title'],
                description=row['description'],
                created_at=row['created_at'],
                expires_at=row['expires_at'],
                is_active=bool(row['is_active']),
                click_count=row['click_count']
            )
            for row in results
        ]

    async def increment_click_count(self, url_id: int) -> bool:
        """Increment the click count for a URL."""
        query = "UPDATE urls SET click_count = click_count + 1 WHERE id = ?"

        rows_affected = await self.database.execute_query(query, (url_id,))
        return rows_affected > 0

    async def deactivate_url(self, url_id: int) -> bool:
        """Deactivate a URL (soft delete)."""
        query = "UPDATE urls SET is_active = FALSE WHERE id = ?"

        rows_affected = await self.database.execute_query(query, (url_id,))
        return rows_affected > 0

    async def update_url(
        self,
        url_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Update URL metadata."""
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if expires_at is not None:
            updates.append("expires_at = ?")
            params.append(expires_at)

        if not updates:
            return False

        params.append(url_id)
        query = f"UPDATE urls SET {', '.join(updates)} WHERE id = ?"

        rows_affected = await self.database.execute_query(query, params)
        return rows_affected > 0

    async def _generate_unique_code(self, max_attempts: int = 10) -> str:
        """Generate a unique short code."""
        for _ in range(max_attempts):
            code = self._generate_random_code()

            # Check if code already exists
            existing = await self.get_url_by_code(code)
            if not existing:
                return code

        # If we can't find a unique code, use a longer one
        return self._generate_random_code(length=8)

    def _generate_random_code(self, length: Optional[int] = None) -> str:
        """Generate a random short code."""
        if length is None:
            length = self.default_code_length

        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    async def _validate_custom_code(self, custom_code: str) -> str:
        """Validate and ensure custom code is unique."""
        # Basic validation
        if not custom_code or len(custom_code) < 3:
            raise ValueError("Custom code must be at least 3 characters long")

        if not custom_code.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Custom code can only contain letters, numbers, hyphens, and underscores")

        # Check uniqueness
        existing = await self.get_url_by_code(custom_code)
        if existing:
            raise ValueError(f"Custom code '{custom_code}' is already in use")

        return custom_code

    async def get_popular_urls(self, limit: int = 10) -> List[URL]:
        """Get most popular URLs by click count."""
        query = """
            SELECT id, short_code, original_url, user_id, title, description,
                   created_at, expires_at, is_active, click_count
            FROM urls
            WHERE is_active = TRUE
            ORDER BY click_count DESC
            LIMIT ?
        """

        results = await self.database.fetch_all(query, (limit,))

        return [
            URL(
                id=row['id'],
                short_code=row['short_code'],
                original_url=row['original_url'],
                user_id=row['user_id'],
                title=row['title'],
                description=row['description'],
                created_at=row['created_at'],
                expires_at=row['expires_at'],
                is_active=bool(row['is_active']),
                click_count=row['click_count']
            )
            for row in results
        ]