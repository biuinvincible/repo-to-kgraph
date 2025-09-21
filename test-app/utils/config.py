"""
Configuration management for URL shortener application.
Handles environment variables and application settings.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration settings."""

    # Database configuration
    database_url: str
    database_pool_size: int

    # Security configuration
    secret_key: str
    password_salt_rounds: int

    # Application settings
    default_url_expiry_hours: int
    max_custom_code_length: int
    min_custom_code_length: int
    rate_limit_per_hour: int

    # Analytics settings
    analytics_retention_days: int
    enable_geo_tracking: bool

    # Server configuration
    server_host: str
    server_port: int
    debug_mode: bool

    def __init__(self):
        """Initialize configuration from environment variables."""

        # Database configuration
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///urlshortener.db')
        self.database_pool_size = int(os.getenv('DATABASE_POOL_SIZE', '5'))

        # Security configuration
        self.secret_key = os.getenv('SECRET_KEY', self._generate_default_secret_key())
        self.password_salt_rounds = int(os.getenv('PASSWORD_SALT_ROUNDS', '12'))

        # Application settings
        self.default_url_expiry_hours = int(os.getenv('DEFAULT_URL_EXPIRY_HOURS', '0'))  # 0 = no expiry
        self.max_custom_code_length = int(os.getenv('MAX_CUSTOM_CODE_LENGTH', '50'))
        self.min_custom_code_length = int(os.getenv('MIN_CUSTOM_CODE_LENGTH', '3'))
        self.rate_limit_per_hour = int(os.getenv('RATE_LIMIT_PER_HOUR', '100'))

        # Analytics settings
        self.analytics_retention_days = int(os.getenv('ANALYTICS_RETENTION_DAYS', '365'))
        self.enable_geo_tracking = os.getenv('ENABLE_GEO_TRACKING', 'false').lower() == 'true'

        # Server configuration
        self.server_host = os.getenv('SERVER_HOST', 'localhost')
        self.server_port = int(os.getenv('SERVER_PORT', '8000'))
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    def _generate_default_secret_key(self) -> str:
        """Generate a default secret key for development."""
        import secrets
        return secrets.token_urlsafe(32)

    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []

        # Validate database URL
        if not self.database_url:
            errors.append("DATABASE_URL is required")

        # Validate secret key
        if len(self.secret_key) < 16:
            errors.append("SECRET_KEY must be at least 16 characters long")

        # Validate port range
        if not (1 <= self.server_port <= 65535):
            errors.append("SERVER_PORT must be between 1 and 65535")

        # Validate custom code length settings
        if self.min_custom_code_length >= self.max_custom_code_length:
            errors.append("MIN_CUSTOM_CODE_LENGTH must be less than MAX_CUSTOM_CODE_LENGTH")

        if self.min_custom_code_length < 1:
            errors.append("MIN_CUSTOM_CODE_LENGTH must be at least 1")

        # Validate rate limiting
        if self.rate_limit_per_hour < 1:
            errors.append("RATE_LIMIT_PER_HOUR must be at least 1")

        # Validate analytics retention
        if self.analytics_retention_days < 1:
            errors.append("ANALYTICS_RETENTION_DAYS must be at least 1")

        if errors:
            raise ConfigurationError("Configuration validation failed:\n" + "\n".join(errors))

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug_mode

    def get_database_config(self) -> dict:
        """Get database-specific configuration."""
        return {
            'url': self.database_url,
            'pool_size': self.database_pool_size
        }

    def get_security_config(self) -> dict:
        """Get security-specific configuration."""
        return {
            'secret_key': self.secret_key,
            'salt_rounds': self.password_salt_rounds
        }

    def get_server_config(self) -> dict:
        """Get server-specific configuration."""
        return {
            'host': self.server_host,
            'port': self.server_port,
            'debug': self.debug_mode
        }

    def get_url_config(self) -> dict:
        """Get URL-specific configuration."""
        return {
            'default_expiry_hours': self.default_url_expiry_hours,
            'max_custom_code_length': self.max_custom_code_length,
            'min_custom_code_length': self.min_custom_code_length,
            'rate_limit_per_hour': self.rate_limit_per_hour
        }

    def get_analytics_config(self) -> dict:
        """Get analytics-specific configuration."""
        return {
            'retention_days': self.analytics_retention_days,
            'enable_geo_tracking': self.enable_geo_tracking
        }

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {
            'database': self.get_database_config(),
            'server': self.get_server_config(),
            'url': self.get_url_config(),
            'analytics': self.get_analytics_config(),
            'debug_mode': self.debug_mode
        }

        if include_sensitive:
            config_dict['security'] = self.get_security_config()
        else:
            config_dict['security'] = {
                'secret_key': '[HIDDEN]',
                'salt_rounds': self.password_salt_rounds
            }

        return config_dict

    @classmethod
    def from_env_file(cls, env_file_path: str) -> 'Config':
        """Load configuration from environment file."""
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

        return cls()

    def save_to_env_file(self, env_file_path: str) -> None:
        """Save current configuration to environment file."""
        env_vars = {
            'DATABASE_URL': self.database_url,
            'DATABASE_POOL_SIZE': str(self.database_pool_size),
            'SECRET_KEY': self.secret_key,
            'PASSWORD_SALT_ROUNDS': str(self.password_salt_rounds),
            'DEFAULT_URL_EXPIRY_HOURS': str(self.default_url_expiry_hours),
            'MAX_CUSTOM_CODE_LENGTH': str(self.max_custom_code_length),
            'MIN_CUSTOM_CODE_LENGTH': str(self.min_custom_code_length),
            'RATE_LIMIT_PER_HOUR': str(self.rate_limit_per_hour),
            'ANALYTICS_RETENTION_DAYS': str(self.analytics_retention_days),
            'ENABLE_GEO_TRACKING': str(self.enable_geo_tracking).lower(),
            'SERVER_HOST': self.server_host,
            'SERVER_PORT': str(self.server_port),
            'DEBUG_MODE': str(self.debug_mode).lower()
        }

        with open(env_file_path, 'w') as f:
            f.write("# URL Shortener Configuration\n")
            f.write("# Generated automatically - modify with caution\n\n")

            f.write("# Database Configuration\n")
            f.write(f"DATABASE_URL={env_vars['DATABASE_URL']}\n")
            f.write(f"DATABASE_POOL_SIZE={env_vars['DATABASE_POOL_SIZE']}\n\n")

            f.write("# Security Configuration\n")
            f.write(f"SECRET_KEY={env_vars['SECRET_KEY']}\n")
            f.write(f"PASSWORD_SALT_ROUNDS={env_vars['PASSWORD_SALT_ROUNDS']}\n\n")

            f.write("# Application Settings\n")
            f.write(f"DEFAULT_URL_EXPIRY_HOURS={env_vars['DEFAULT_URL_EXPIRY_HOURS']}\n")
            f.write(f"MAX_CUSTOM_CODE_LENGTH={env_vars['MAX_CUSTOM_CODE_LENGTH']}\n")
            f.write(f"MIN_CUSTOM_CODE_LENGTH={env_vars['MIN_CUSTOM_CODE_LENGTH']}\n")
            f.write(f"RATE_LIMIT_PER_HOUR={env_vars['RATE_LIMIT_PER_HOUR']}\n\n")

            f.write("# Analytics Settings\n")
            f.write(f"ANALYTICS_RETENTION_DAYS={env_vars['ANALYTICS_RETENTION_DAYS']}\n")
            f.write(f"ENABLE_GEO_TRACKING={env_vars['ENABLE_GEO_TRACKING']}\n\n")

            f.write("# Server Configuration\n")
            f.write(f"SERVER_HOST={env_vars['SERVER_HOST']}\n")
            f.write(f"SERVER_PORT={env_vars['SERVER_PORT']}\n")
            f.write(f"DEBUG_MODE={env_vars['DEBUG_MODE']}\n")


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """Manager for handling multiple configuration profiles."""

    def __init__(self):
        self.profiles = {}
        self.current_profile = 'default'

    def add_profile(self, name: str, config: Config) -> None:
        """Add a configuration profile."""
        self.profiles[name] = config

    def set_profile(self, name: str) -> None:
        """Set the active configuration profile."""
        if name not in self.profiles:
            raise ConfigurationError(f"Profile '{name}' not found")
        self.current_profile = name

    def get_config(self, profile_name: Optional[str] = None) -> Config:
        """Get configuration for specified or current profile."""
        profile = profile_name or self.current_profile

        if profile not in self.profiles:
            raise ConfigurationError(f"Profile '{profile}' not found")

        return self.profiles[profile]

    def load_profiles_from_directory(self, directory: str) -> None:
        """Load configuration profiles from directory."""
        if not os.path.exists(directory):
            return

        for filename in os.listdir(directory):
            if filename.endswith('.env'):
                profile_name = filename[:-4]  # Remove .env extension
                env_file_path = os.path.join(directory, filename)

                try:
                    config = Config.from_env_file(env_file_path)
                    config.validate()
                    self.add_profile(profile_name, config)
                except Exception as e:
                    print(f"Warning: Failed to load profile '{profile_name}': {e}")

    def get_available_profiles(self) -> list:
        """Get list of available configuration profiles."""
        return list(self.profiles.keys())