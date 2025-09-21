# URL Shortener Service

A complete URL shortening service with analytics and user management.

## Features

- **URL Shortening**: Create short codes for long URLs with optional custom codes
- **User Management**: User accounts with role-based access control (User, Moderator, Admin)
- **Analytics**: Comprehensive click tracking and statistics
- **Security**: Password hashing, JWT tokens, and input validation
- **Database**: SQLite with async operations
- **Expiration**: Optional URL expiration dates

## Architecture

The application follows a clean architecture pattern with separate layers:

- **Models**: Data structures for URLs, users, and analytics
- **Services**: Business logic for URL management, user operations, and analytics
- **Utils**: Database connectivity, configuration, and security utilities

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The application will:
- Create necessary database tables
- Set up a default admin user (username: admin, password: admin123)
- Run a demo showing URL creation and analytics

## Usage Examples

### Creating Short URLs
```python
# Basic URL shortening
url = await app.create_short_url("https://www.example.com")

# With custom code and expiration
url = await app.create_short_url(
    "https://www.example.com",
    custom_code="example",
    expires_hours=24
)
```

### Resolving URLs
```python
original_url = await app.resolve_short_url("abc123", {
    "ip": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "referer": "https://google.com"
})
```

### Analytics
```python
stats = await app.get_url_analytics("abc123")
print(f"Total clicks: {stats.total_clicks}")
print(f"Unique visitors: {stats.unique_visitors}")
```

## Configuration

Configuration is handled through environment variables:

- `DATABASE_URL`: Database connection string (default: sqlite:///urlshortener.db)
- `SECRET_KEY`: Secret key for JWT tokens and security
- `DEBUG_MODE`: Enable debug mode (default: false)
- `RATE_LIMIT_PER_HOUR`: Rate limiting for URL creation (default: 100)

## Security Features

- **Password Hashing**: bcrypt with configurable salt rounds
- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Protection against abuse
- **URL Validation**: Safety checks for malicious URLs

## Database Schema

### Users Table
- User accounts with roles and authentication
- Tracks creation date and last login

### URLs Table
- Short codes mapped to original URLs
- User ownership and metadata
- Click tracking and expiration

### URL Clicks Table
- Detailed analytics for each click
- IP address, user agent, and referrer tracking
- Geographic information (when available)

## Development

The codebase is structured for maintainability and testing:

- Type hints throughout
- Async/await patterns
- Error handling and logging
- Clean separation of concerns

## Testing

Run tests with:
```bash
pytest
```

For coverage reports:
```bash
pytest --cov=.
```