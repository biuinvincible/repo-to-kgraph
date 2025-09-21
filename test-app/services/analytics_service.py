"""
Analytics service for tracking URL usage and generating insights.
Handles click tracking, statistics generation, and reporting.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from models.url import UrlStats
from utils.database import Database


class AnalyticsService:
    """Service for managing analytics and tracking."""

    def __init__(self, database: Database):
        self.database = database

    async def track_click(
        self,
        url_id: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        referer: Optional[str] = None,
        country: Optional[str] = None
    ) -> None:
        """Track a URL click with metadata."""

        # Get country from IP if not provided
        if ip_address and not country:
            country = self._get_country_from_ip(ip_address)

        # Insert click record
        query = """
            INSERT INTO url_clicks (url_id, ip_address, user_agent, referer, country)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (url_id, ip_address, user_agent, referer, country)

        await self.database.execute_query(query, params)

    async def get_url_stats(self, url_id: int, days: int = 30) -> UrlStats:
        """Get comprehensive statistics for a URL."""

        # Base query for recent clicks
        base_query = """
            FROM url_clicks
            WHERE url_id = ? AND clicked_at >= ?
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        base_params = (url_id, cutoff_date)

        # Get total clicks
        total_clicks = await self._get_total_clicks(url_id, cutoff_date)

        # Get unique visitors
        unique_visitors = await self._get_unique_visitors(url_id, cutoff_date)

        # Get clicks by country
        clicks_by_country = await self._get_clicks_by_country(url_id, cutoff_date)

        # Get clicks by hour
        clicks_by_hour = await self._get_clicks_by_hour(url_id, cutoff_date)

        # Get clicks by day
        clicks_by_day = await self._get_clicks_by_day(url_id, cutoff_date)

        # Get top referers
        top_referers = await self._get_top_referers(url_id, cutoff_date)

        # Calculate average clicks per day
        avg_clicks_per_day = total_clicks / days if days > 0 else 0

        return UrlStats(
            url_id=url_id,
            total_clicks=total_clicks,
            unique_visitors=unique_visitors,
            clicks_by_country=clicks_by_country,
            clicks_by_hour=clicks_by_hour,
            clicks_by_day=clicks_by_day,
            top_referers=top_referers,
            avg_clicks_per_day=avg_clicks_per_day
        )

    async def log_event(
        self,
        event_type: str,
        url_id: Optional[int] = None,
        user_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a general analytics event."""

        # For this implementation, we'll just log to a simple events table
        # In production, you might use a more sophisticated event tracking system

        query = """
            INSERT OR IGNORE INTO events (event_type, url_id, user_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """

        # Convert metadata to JSON string if provided
        metadata_json = None
        if metadata:
            import json
            metadata_json = json.dumps(metadata)

        params = (event_type, url_id, user_id, metadata_json, datetime.utcnow())

        try:
            await self.database.execute_query(query, params)
        except Exception:
            # Create events table if it doesn't exist
            await self._create_events_table()
            await self.database.execute_query(query, params)

    async def get_global_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get global analytics statistics."""

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Total URLs created
        total_urls_query = "SELECT COUNT(*) as count FROM urls WHERE created_at >= ?"
        total_urls = await self.database.fetch_one(total_urls_query, (cutoff_date,))

        # Total clicks
        total_clicks_query = "SELECT COUNT(*) as count FROM url_clicks WHERE clicked_at >= ?"
        total_clicks = await self.database.fetch_one(total_clicks_query, (cutoff_date,))

        # Active URLs
        active_urls_query = "SELECT COUNT(*) as count FROM urls WHERE is_active = TRUE"
        active_urls = await self.database.fetch_one(active_urls_query)

        # Most popular URLs
        popular_urls_query = """
            SELECT u.short_code, u.original_url, COUNT(c.id) as click_count
            FROM urls u
            LEFT JOIN url_clicks c ON u.id = c.url_id AND c.clicked_at >= ?
            WHERE u.is_active = TRUE
            GROUP BY u.id
            ORDER BY click_count DESC
            LIMIT 10
        """
        popular_urls = await self.database.fetch_all(popular_urls_query, (cutoff_date,))

        return {
            'period_days': days,
            'total_urls_created': total_urls['count'],
            'total_clicks': total_clicks['count'],
            'active_urls': active_urls['count'],
            'popular_urls': [
                {
                    'short_code': row['short_code'],
                    'original_url': row['original_url'],
                    'clicks': row['click_count']
                }
                for row in popular_urls
            ]
        }

    async def _get_total_clicks(self, url_id: int, cutoff_date: datetime) -> int:
        """Get total clicks for URL since cutoff date."""
        query = "SELECT COUNT(*) as count FROM url_clicks WHERE url_id = ? AND clicked_at >= ?"
        result = await self.database.fetch_one(query, (url_id, cutoff_date))
        return result['count']

    async def _get_unique_visitors(self, url_id: int, cutoff_date: datetime) -> int:
        """Get unique visitors for URL since cutoff date."""
        query = """
            SELECT COUNT(DISTINCT ip_address) as count
            FROM url_clicks
            WHERE url_id = ? AND clicked_at >= ? AND ip_address IS NOT NULL
        """
        result = await self.database.fetch_one(query, (url_id, cutoff_date))
        return result['count']

    async def _get_clicks_by_country(self, url_id: int, cutoff_date: datetime) -> Dict[str, int]:
        """Get clicks grouped by country."""
        query = """
            SELECT country, COUNT(*) as count
            FROM url_clicks
            WHERE url_id = ? AND clicked_at >= ? AND country IS NOT NULL
            GROUP BY country
            ORDER BY count DESC
        """
        results = await self.database.fetch_all(query, (url_id, cutoff_date))
        return {row['country']: row['count'] for row in results}

    async def _get_clicks_by_hour(self, url_id: int, cutoff_date: datetime) -> Dict[int, int]:
        """Get clicks grouped by hour of day."""
        query = """
            SELECT strftime('%H', clicked_at) as hour, COUNT(*) as count
            FROM url_clicks
            WHERE url_id = ? AND clicked_at >= ?
            GROUP BY hour
            ORDER BY hour
        """
        results = await self.database.fetch_all(query, (url_id, cutoff_date))
        return {int(row['hour']): row['count'] for row in results}

    async def _get_clicks_by_day(self, url_id: int, cutoff_date: datetime) -> Dict[str, int]:
        """Get clicks grouped by day."""
        query = """
            SELECT DATE(clicked_at) as day, COUNT(*) as count
            FROM url_clicks
            WHERE url_id = ? AND clicked_at >= ?
            GROUP BY day
            ORDER BY day
        """
        results = await self.database.fetch_all(query, (url_id, cutoff_date))
        return {row['day']: row['count'] for row in results}

    async def _get_top_referers(self, url_id: int, cutoff_date: datetime) -> Dict[str, int]:
        """Get top referers for URL."""
        query = """
            SELECT referer, COUNT(*) as count
            FROM url_clicks
            WHERE url_id = ? AND clicked_at >= ? AND referer IS NOT NULL
            GROUP BY referer
            ORDER BY count DESC
            LIMIT 10
        """
        results = await self.database.fetch_all(query, (url_id, cutoff_date))
        return {row['referer']: row['count'] for row in results}

    def _get_country_from_ip(self, ip_address: str) -> str:
        """Get country from IP address (simplified implementation)."""
        # In a real implementation, you would use a GeoIP service
        # For this demo, we'll return a placeholder
        return "Unknown"

    async def _create_events_table(self) -> None:
        """Create events table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                url_id INTEGER,
                user_id INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (url_id) REFERENCES urls(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """
        await self.database.execute_query(query)

    async def generate_report(self, url_id: int, format: str = "summary") -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""

        stats = await self.get_url_stats(url_id)

        if format == "summary":
            return {
                'total_clicks': stats.total_clicks,
                'unique_visitors': stats.unique_visitors,
                'avg_clicks_per_day': round(stats.avg_clicks_per_day, 2),
                'top_country': max(stats.clicks_by_country.items(), key=lambda x: x[1])[0] if stats.clicks_by_country else None,
                'peak_hour': stats.get_peak_hour()
            }

        elif format == "detailed":
            return {
                'overview': {
                    'total_clicks': stats.total_clicks,
                    'unique_visitors': stats.unique_visitors,
                    'avg_clicks_per_day': stats.avg_clicks_per_day
                },
                'geographic': stats.clicks_by_country,
                'temporal': {
                    'by_hour': stats.clicks_by_hour,
                    'by_day': stats.clicks_by_day
                },
                'referrers': stats.top_referers
            }

        return stats.__dict__