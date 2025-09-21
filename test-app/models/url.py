"""
URL model for URL shortener service.
Represents shortened URLs with metadata and analytics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class URL:
    """Represents a shortened URL with all associated metadata."""

    id: int
    short_code: str
    original_url: str
    user_id: Optional[int]
    title: Optional[str]
    description: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    click_count: int

    def is_expired(self) -> bool:
        """Check if URL has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def can_be_accessed(self) -> bool:
        """Check if URL can be accessed (active and not expired)."""
        return self.is_active and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Convert URL to dictionary representation."""
        return {
            'id': self.id,
            'short_code': self.short_code,
            'original_url': self.original_url,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'click_count': self.click_count
        }


@dataclass
class UrlStats:
    """Analytics statistics for a URL."""

    url_id: int
    total_clicks: int
    unique_visitors: int
    clicks_by_country: Dict[str, int]
    clicks_by_hour: Dict[int, int]
    clicks_by_day: Dict[str, int]
    top_referers: Dict[str, int]
    avg_clicks_per_day: float

    def get_top_countries(self, limit: int = 5) -> Dict[str, int]:
        """Get top countries by click count."""
        sorted_countries = sorted(
            self.clicks_by_country.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_countries[:limit])

    def get_peak_hour(self) -> int:
        """Get hour with most clicks."""
        if not self.clicks_by_hour:
            return 0
        return max(self.clicks_by_hour.items(), key=lambda x: x[1])[0]

    def calculate_click_rate(self, days: int) -> float:
        """Calculate click rate per day over specified period."""
        if days <= 0:
            return 0.0
        return self.total_clicks / days