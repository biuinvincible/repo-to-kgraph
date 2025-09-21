#!/usr/bin/env python3
"""
URL Shortener Service - Main Application
A complete URL shortening service with analytics and user management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from models.url import URL, UrlStats
from models.user import User, UserRole
from services.url_service import UrlService
from services.user_service import UserService
from services.analytics_service import AnalyticsService
from utils.database import Database
from utils.config import Config
from utils.security import SecurityManager


class UrlShortenerApp:
    """Main application class for URL shortener service."""

    def __init__(self):
        """Initialize the application with all services."""
        self.config = Config()
        self.database = Database(self.config.database_url)
        self.security = SecurityManager(self.config.secret_key)

        # Initialize services
        self.url_service = UrlService(self.database, self.security)
        self.user_service = UserService(self.database, self.security)
        self.analytics_service = AnalyticsService(self.database)

        self.logger = logging.getLogger(__name__)

    async def startup(self):
        """Start the application and initialize all components."""
        try:
            self.logger.info("Starting URL Shortener Service...")

            # Connect to database
            await self.database.connect()

            # Initialize tables
            await self._init_database_schema()

            # Create default admin user if not exists
            await self._create_default_admin()

            self.logger.info("Application started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            raise

    async def shutdown(self):
        """Cleanup and shutdown the application."""
        try:
            self.logger.info("Shutting down URL Shortener Service...")

            # Close database connection
            await self.database.disconnect()

            self.logger.info("Application shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _init_database_schema(self):
        """Initialize database tables."""
        try:
            # Create tables if they don't exist
            await self.database.execute_query("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            await self.database.execute_query("""
                CREATE TABLE IF NOT EXISTS urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    short_code TEXT UNIQUE NOT NULL,
                    original_url TEXT NOT NULL,
                    user_id INTEGER,
                    title TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    click_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            await self.database.execute_query("""
                CREATE TABLE IF NOT EXISTS url_clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url_id INTEGER NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    referer TEXT,
                    country TEXT,
                    clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (url_id) REFERENCES urls(id)
                )
            """)

            self.logger.info("Database schema initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}")
            raise

    async def _create_default_admin(self):
        """Create default admin user if no users exist."""
        try:
            # Check if any users exist
            users = await self.user_service.get_all_users(limit=1)

            if not users:
                # Create default admin
                admin_user = await self.user_service.create_user(
                    username="admin",
                    email="admin@urlshortener.com",
                    password="admin123",
                    role=UserRole.ADMIN
                )

                self.logger.info(f"Created default admin user: {admin_user.username}")

        except Exception as e:
            self.logger.error(f"Failed to create default admin: {e}")

    async def create_short_url(
        self,
        original_url: str,
        user_id: Optional[int] = None,
        custom_code: Optional[str] = None,
        expires_hours: Optional[int] = None
    ) -> URL:
        """Create a new short URL."""
        try:
            # Validate URL format
            if not self._is_valid_url(original_url):
                raise ValueError("Invalid URL format")

            # Set expiration if specified
            expires_at = None
            if expires_hours:
                expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

            # Create URL through service
            url = await self.url_service.create_url(
                original_url=original_url,
                user_id=user_id,
                custom_code=custom_code,
                expires_at=expires_at
            )

            # Log analytics event
            await self.analytics_service.log_event(
                event_type="url_created",
                url_id=url.id,
                user_id=user_id
            )

            self.logger.info(f"Created short URL: {url.short_code} -> {original_url}")
            return url

        except Exception as e:
            self.logger.error(f"Failed to create short URL: {e}")
            raise

    async def resolve_short_url(self, short_code: str, client_info: dict = None) -> str:
        """Resolve a short code to original URL and track analytics."""
        try:
            # Get URL by short code
            url = await self.url_service.get_url_by_code(short_code)

            if not url:
                raise ValueError(f"Short URL not found: {short_code}")

            # Check if URL is active and not expired
            if not url.is_active:
                raise ValueError("URL has been deactivated")

            if url.expires_at and url.expires_at < datetime.utcnow():
                raise ValueError("URL has expired")

            # Track click analytics
            await self.analytics_service.track_click(
                url_id=url.id,
                ip_address=client_info.get("ip") if client_info else None,
                user_agent=client_info.get("user_agent") if client_info else None,
                referer=client_info.get("referer") if client_info else None
            )

            # Update click count
            await self.url_service.increment_click_count(url.id)

            self.logger.info(f"Resolved short URL: {short_code} -> {url.original_url}")
            return url.original_url

        except Exception as e:
            self.logger.error(f"Failed to resolve short URL {short_code}: {e}")
            raise

    async def get_url_analytics(self, short_code: str, user_id: Optional[int] = None) -> UrlStats:
        """Get analytics for a short URL."""
        try:
            # Get URL
            url = await self.url_service.get_url_by_code(short_code)

            if not url:
                raise ValueError(f"Short URL not found: {short_code}")

            # Check permissions (users can only see their own URLs)
            if user_id and url.user_id != user_id:
                user = await self.user_service.get_user_by_id(user_id)
                if not user or user.role != UserRole.ADMIN:
                    raise PermissionError("Access denied to URL analytics")

            # Get analytics data
            stats = await self.analytics_service.get_url_stats(url.id)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get analytics for {short_code}: {e}")
            raise

    async def delete_short_url(self, short_code: str, user_id: int) -> bool:
        """Delete a short URL (soft delete by deactivating)."""
        try:
            # Get URL
            url = await self.url_service.get_url_by_code(short_code)

            if not url:
                raise ValueError(f"Short URL not found: {short_code}")

            # Check permissions
            user = await self.user_service.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")

            if url.user_id != user_id and user.role != UserRole.ADMIN:
                raise PermissionError("Access denied to delete this URL")

            # Deactivate URL
            success = await self.url_service.deactivate_url(url.id)

            if success:
                # Log analytics event
                await self.analytics_service.log_event(
                    event_type="url_deleted",
                    url_id=url.id,
                    user_id=user_id
                )

                self.logger.info(f"Deleted short URL: {short_code}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete short URL {short_code}: {e}")
            raise

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        import re

        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return url_pattern.match(url) is not None


async def main():
    """Main application entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and start application
    app = UrlShortenerApp()

    try:
        await app.startup()

        # Demo: Create some sample URLs
        print("\n🔗 URL Shortener Service Demo")
        print("=" * 40)

        # Create sample URLs
        url1 = await app.create_short_url("https://www.google.com", expires_hours=24)
        print(f"Created: {url1.short_code} -> {url1.original_url}")

        url2 = await app.create_short_url("https://github.com", custom_code="github")
        print(f"Created: {url2.short_code} -> {url2.original_url}")

        # Test resolution
        resolved = await app.resolve_short_url(url1.short_code, {
            "ip": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Test Browser)"
        })
        print(f"Resolved: {url1.short_code} -> {resolved}")

        # Get analytics
        stats = await app.get_url_analytics(url1.short_code)
        print(f"Analytics: {url1.short_code} has {stats.total_clicks} clicks")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())