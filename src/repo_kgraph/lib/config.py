"""
Configuration management for the repository knowledge graph system.

Handles loading and validation of configuration from environment variables,
configuration files, and default values.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
import yaml
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""

    # Neo4j configuration
    neo4j_uri: str = Field(default="neo4j://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_max_connection_lifetime: int = Field(default=30, description="Neo4j connection lifetime in seconds")
    neo4j_max_connection_pool_size: int = Field(default=50, description="Neo4j max connection pool size")

    # Chroma configuration
    chroma_db_path: str = Field(default="./chroma_db", description="Path to ChromaDB storage")
    chroma_collection_prefix: str = Field(default="repo_", description="Prefix for ChromaDB collections")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    embedding_provider: str = Field(default="ollama", description="Embedding provider")
    model_name: str = Field(default="embeddinggemma", description="Embedding model name")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    batch_size: int = Field(default=32, ge=1, le=1000, description="Batch size for embedding generation")
    max_text_length: int = Field(default=8192, ge=100, description="Maximum text length for embeddings")

    @field_validator("embedding_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["ollama", "sentence_transformers", "openai"]
        if v not in valid_providers:
            raise ValueError(f"Embedding provider must be one of {valid_providers}")
        return v


class ParsingConfig(BaseModel):
    """Configuration for code parsing."""

    max_file_size_mb: float = Field(default=10.0, ge=0.1, description="Maximum file size to parse in MB")
    max_concurrent_files: int = Field(default=10, ge=1, le=100, description="Maximum concurrent files to parse")
    default_exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.pyc", "__pycache__/*", "node_modules/*", ".git/*",
            "*.min.js", "*.min.css", "build/*", "dist/*", ".venv/*", "venv/*"
        ],
        description="Default file patterns to exclude from parsing"
    )
    timeout_seconds: int = Field(default=300, ge=10, description="Parsing timeout in seconds")


class RetrievalConfig(BaseModel):
    """Configuration for context retrieval."""

    default_max_results: int = Field(default=20, ge=1, le=100, description="Default maximum results")
    default_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Default confidence threshold")
    graph_traversal_depth: int = Field(default=2, ge=1, le=5, description="Maximum graph traversal depth")
    similarity_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for similarity scores")
    graph_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for graph scores")
    query_timeout_seconds: int = Field(default=30, ge=5, description="Query timeout in seconds")


class APIConfig(BaseModel):
    """Configuration for API server."""

    host: str = Field(default="localhost", description="API server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins")
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class Config(BaseModel):
    """Main configuration class combining all configuration sections."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Global settings
    data_dir: str = Field(default="./data", description="Base data directory")
    log_dir: str = Field(default="./logs", description="Log directory")
    temp_dir: str = Field(default="./tmp", description="Temporary directory")

    class Config:
        extra = "forbid"  # Prevent extra fields


class ConfigManager:
    """
    Manager for loading and accessing configuration.

    Supports configuration from environment variables, YAML files,
    and provides defaults with validation.
    """

    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to YAML configuration file
            env_file: Path to .env file for environment variables
        """
        self.config_file = config_file
        self.env_file = env_file
        self._config: Optional[Config] = None

        # Load environment variables
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        elif Path(".env").exists():
            load_dotenv(".env")

    def load_config(self) -> Config:
        """
        Load configuration from all sources.

        Returns:
            Validated configuration object
        """
        if self._config is not None:
            return self._config

        # Start with default config
        config_data = {}

        # Load from YAML file if provided
        if self.config_file and Path(self.config_file).exists():
            config_data = self._load_yaml_config(self.config_file)

        # Override with environment variables
        env_overrides = self._load_env_config()
        config_data = self._merge_config(config_data, env_overrides)

        # Create and validate config
        self._config = Config(**config_data)

        # Create directories if needed
        self._ensure_directories()

        logger.info("Configuration loaded successfully")
        return self._config

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config_data or {}
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
            return {}

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Database configuration
        db_config = {}
        if os.getenv("NEO4J_URI"):
            db_config["neo4j_uri"] = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USERNAME"):
            db_config["neo4j_username"] = os.getenv("NEO4J_USERNAME")
        if os.getenv("NEO4J_PASSWORD"):
            db_config["neo4j_password"] = os.getenv("NEO4J_PASSWORD")
        if os.getenv("NEO4J_DATABASE"):
            db_config["neo4j_database"] = os.getenv("NEO4J_DATABASE")
        if os.getenv("CHROMA_DB_PATH"):
            db_config["chroma_db_path"] = os.getenv("CHROMA_DB_PATH")

        if db_config:
            env_config["database"] = db_config

        # Embedding configuration
        embedding_config = {}
        if os.getenv("EMBEDDING_PROVIDER"):
            embedding_config["embedding_provider"] = os.getenv("EMBEDDING_PROVIDER")
        if os.getenv("EMBEDDING_MODEL"):
            embedding_config["model_name"] = os.getenv("EMBEDDING_MODEL")
        if os.getenv("OPENAI_API_KEY"):
            embedding_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("EMBEDDING_BATCH_SIZE"):
            try:
                embedding_config["batch_size"] = int(os.getenv("EMBEDDING_BATCH_SIZE"))
            except ValueError:
                logger.warning("Invalid EMBEDDING_BATCH_SIZE, using default")

        if embedding_config:
            env_config["embedding"] = embedding_config

        # API configuration
        api_config = {}
        if os.getenv("API_HOST"):
            api_config["host"] = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            try:
                api_config["port"] = int(os.getenv("API_PORT"))
            except ValueError:
                logger.warning("Invalid API_PORT, using default")
        if os.getenv("API_DEBUG"):
            api_config["debug"] = os.getenv("API_DEBUG").lower() in ("true", "1", "yes")
        if os.getenv("LOG_LEVEL"):
            api_config["log_level"] = os.getenv("LOG_LEVEL")

        if api_config:
            env_config["api"] = api_config

        # Global settings
        if os.getenv("DATA_DIR"):
            env_config["data_dir"] = os.getenv("DATA_DIR")
        if os.getenv("LOG_DIR"):
            env_config["log_dir"] = os.getenv("LOG_DIR")
        if os.getenv("TEMP_DIR"):
            env_config["temp_dir"] = os.getenv("TEMP_DIR")

        return env_config

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        if self._config:
            directories = [
                self._config.data_dir,
                self._config.log_dir,
                self._config.temp_dir,
                self._config.database.chroma_db_path
            ]

            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)

    def get_config(self) -> Config:
        """Get the current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config

    def reload_config(self) -> Config:
        """Reload configuration from sources."""
        self._config = None
        return self.load_config()

    def save_config(self, config_file: str) -> None:
        """Save current configuration to YAML file."""
        if self._config is None:
            raise ValueError("No configuration loaded")

        config_dict = self._config.model_dump()

        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_file}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    config_file: Optional[str] = None,
    env_file: Optional[str] = None
) -> ConfigManager:
    """
    Get global configuration manager instance.

    Args:
        config_file: Path to YAML configuration file
        env_file: Path to .env file

    Returns:
        ConfigManager instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_file, env_file)

    return _config_manager


def get_config() -> Config:
    """Get the current configuration."""
    return get_config_manager().get_config()


def setup_logging(config: Optional[Config] = None) -> None:
    """Setup logging based on configuration."""
    if config is None:
        config = get_config()

    # Ensure log directory exists
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging configuration
    log_level = getattr(logging, config.api.log_level, logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / "repo_kgraph.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    logger.info("Logging setup completed")


# Example configuration template
CONFIG_TEMPLATE = """
# Repository Knowledge Graph Configuration

database:
  neo4j_uri: "neo4j://localhost:7687"
  neo4j_username: "neo4j"
  neo4j_password: "password"
  neo4j_database: "neo4j"
  chroma_db_path: "./chroma_db"

embedding:
  embedding_provider: "ollama"  # or "sentence_transformers" or "openai"
  model_name: "embeddinggemma"
  # openai_api_key: "your-openai-key"  # Required if using OpenAI
  batch_size: 32
  max_text_length: 8192

parsing:
  max_file_size_mb: 10.0
  max_concurrent_files: 10
  timeout_seconds: 300

retrieval:
  default_max_results: 20
  default_confidence_threshold: 0.3
  graph_traversal_depth: 2
  similarity_weight: 0.7
  graph_weight: 0.3

api:
  host: "localhost"
  port: 8000
  debug: false
  log_level: "INFO"

# Global directories
data_dir: "./data"
log_dir: "./logs"
temp_dir: "./tmp"
"""


def create_config_template(file_path: str) -> None:
    """Create a configuration template file."""
    with open(file_path, 'w') as f:
        f.write(CONFIG_TEMPLATE)
    logger.info(f"Configuration template created at {file_path}")


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    config = config_manager.load_config()

    print("Configuration loaded:")
    print(f"Neo4j URI: {config.database.neo4j_uri}")
    print(f"Embedding Provider: {config.embedding.embedding_provider}")
    print(f"API Host: {config.api.host}:{config.api.port}")
    print(f"Log Level: {config.api.log_level}")