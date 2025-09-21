"""
Base CLI infrastructure and common utilities.

Provides common functionality for all CLI commands including
configuration, logging, and error handling.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

from repo_kgraph.lib.config import get_config_manager, setup_logging, Config
from repo_kgraph.lib.database import get_database_manager, DatabaseManager
from repo_kgraph.services.parser import CodeParser
from repo_kgraph.services.graph_builder import GraphBuilder
from repo_kgraph.services.embedding import EmbeddingService
from repo_kgraph.services.repository_manager import RepositoryManager
from repo_kgraph.services.retriever import ContextRetriever
from repo_kgraph.services.query_processor import QueryProcessor


# Rich console for formatting output
console = Console()
logger = logging.getLogger(__name__)


class CLIError(Exception):
    """Base exception for CLI errors."""
    pass


class CLIContext:
    """
    Context object holding shared CLI state and services.

    Manages configuration, database connections, and service instances
    across different CLI commands.
    """

    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize CLI context.

        Args:
            config_file: Path to configuration file
            env_file: Path to environment file
        """
        self.config_file = config_file
        self.env_file = env_file
        self._config: Optional[Config] = None
        self._database_manager: Optional[DatabaseManager] = None
        self._services: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize CLI context with configuration and services."""
        try:
            # Load configuration
            config_manager = get_config_manager(self.config_file, self.env_file)
            self._config = config_manager.load_config()

            # Setup logging
            setup_logging(self._config)

            # Initialize database connections
            self._database_manager = await get_database_manager(self._config)

            # Initialize services
            await self._initialize_services()

            logger.info("CLI context initialized successfully")

        except Exception as e:
            console.print(f"[red]Failed to initialize CLI context: {e}[/red]")
            raise CLIError(f"Initialization failed: {e}")

    async def _initialize_services(self) -> None:
        """Initialize all services."""
        if not self._config or not self._database_manager:
            raise CLIError("Configuration or database manager not available")

        try:
            # Code parser
            parser = CodeParser(max_file_size_mb=self._config.parsing.max_file_size_mb)
            self._services["parser"] = parser

            # Graph builder
            graph_builder = GraphBuilder(
                uri=self._config.database.neo4j_uri,
                username=self._config.database.neo4j_username,
                password=self._config.database.neo4j_password,
                database=self._config.database.neo4j_database
            )
            self._services["graph_builder"] = graph_builder

            # Embedding service
            embedding_service = EmbeddingService(
                model_name=self._config.embedding.model_name,
                chroma_db_path=self._config.database.chroma_db_path,
                batch_size=self._config.embedding.batch_size,
                max_text_length=self._config.embedding.max_text_length,
                device=self._config.embedding.device
            )
            self._services["embedding_service"] = embedding_service

            # Context retriever
            context_retriever = ContextRetriever(
                embedding_service=embedding_service,
                graph_builder=graph_builder,
                default_max_results=self._config.retrieval.default_max_results,
                default_confidence_threshold=self._config.retrieval.default_confidence_threshold,
                graph_traversal_depth=self._config.retrieval.graph_traversal_depth,
                similarity_weight=self._config.retrieval.similarity_weight,
                graph_weight=self._config.retrieval.graph_weight
            )
            self._services["context_retriever"] = context_retriever

            # Query processor
            query_processor = QueryProcessor(
                context_retriever=context_retriever,
                embedding_service=embedding_service,
                default_max_results=self._config.retrieval.default_max_results,
                default_confidence_threshold=self._config.retrieval.default_confidence_threshold,
                query_timeout_seconds=self._config.retrieval.query_timeout_seconds
            )
            self._services["query_processor"] = query_processor

            # Repository manager
            repository_manager = RepositoryManager(
                parser=parser,
                graph_builder=graph_builder,
                embedding_service=embedding_service,
                max_concurrent_files=self._config.parsing.max_concurrent_files,
                batch_size=1000
            )
            self._services["repository_manager"] = repository_manager

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise

    @property
    def config(self) -> Config:
        """Get configuration."""
        if not self._config:
            raise CLIError("Configuration not loaded")
        return self._config

    @property
    def database_manager(self) -> DatabaseManager:
        """Get database manager."""
        if not self._database_manager:
            raise CLIError("Database manager not initialized")
        return self._database_manager

    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name.

        Args:
            service_name: Name of the service

        Returns:
            Service instance

        Raises:
            CLIError: If service not found
        """
        service = self._services.get(service_name)
        if not service:
            raise CLIError(f"Service '{service_name}' not available")
        return service

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Cleanup repository manager
            if "repository_manager" in self._services:
                await self._services["repository_manager"].cleanup()

            # Cleanup query processor
            if "query_processor" in self._services:
                await self._services["query_processor"].cleanup()

            # Cleanup embedding service
            if "embedding_service" in self._services:
                await self._services["embedding_service"].cleanup()

            # Close database connections
            if self._database_manager:
                await self._database_manager.close()

            logger.info("CLI context cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global CLI context
_cli_context: Optional[CLIContext] = None


def get_cli_context() -> CLIContext:
    """Get global CLI context."""
    global _cli_context
    if not _cli_context:
        raise CLIError("CLI context not initialized")
    return _cli_context


def set_cli_context(context: CLIContext) -> None:
    """Set global CLI context."""
    global _cli_context
    _cli_context = context


# CLI decorators and utilities
def async_command(f):
    """Decorator to run async functions in CLI commands."""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


def handle_cli_errors(f):
    """Decorator to handle CLI errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except CLIError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(130)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            logger.exception("Unexpected error in CLI command")
            sys.exit(1)
    return wrapper


# Output formatting utilities
def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_table(data: list, headers: list, title: Optional[str] = None) -> None:
    """
    Print data as a formatted table.

    Args:
        data: List of row data
        headers: List of column headers
        title: Optional table title
    """
    table = Table(title=title)

    for header in headers:
        table.add_column(header, justify="left")

    for row in data:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)


def print_panel(content: str, title: Optional[str] = None, style: str = "blue") -> None:
    """
    Print content in a panel.

    Args:
        content: Content to display
        title: Optional panel title
        style: Panel style
    """
    panel = Panel(content, title=title, border_style=style)
    console.print(panel)


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Prompt user for confirmation.

    Args:
        message: Confirmation message
        default: Default response

    Returns:
        True if confirmed, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    response = input(f"{message}{suffix}: ").strip().lower()

    if not response:
        return default

    return response in ["y", "yes", "true", "1"]


class ProgressReporter:
    """Progress reporter for long-running operations."""

    def __init__(self, title: str = "Processing"):
        """
        Initialize progress reporter.

        Args:
            title: Progress title
        """
        self.title = title
        self.progress = None
        self.task_id = None

    def __enter__(self):
        """Enter progress context."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
        )
        self.progress.start()
        self.task_id = self.progress.add_task(self.title, total=100)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit progress context."""
        if self.progress:
            self.progress.stop()

    def update(self, progress: float, description: Optional[str] = None):
        """
        Update progress.

        Args:
            progress: Progress percentage (0-100)
            description: Optional description update
        """
        if self.progress and self.task_id:
            update_kwargs = {"completed": progress}
            if description:
                update_kwargs["description"] = description
            self.progress.update(self.task_id, **update_kwargs)


def validate_repository_path(path: str) -> Path:
    """
    Validate repository path.

    Args:
        path: Repository path string

    Returns:
        Validated Path object

    Raises:
        CLIError: If path is invalid
    """
    repo_path = Path(path).resolve()

    if not repo_path.exists():
        raise CLIError(f"Repository path does not exist: {path}")

    if not repo_path.is_dir():
        raise CLIError(f"Repository path is not a directory: {path}")

    return repo_path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# Common CLI options
def add_common_options(f):
    """Add common CLI options to a command."""
    f = click.option(
        "--config",
        type=click.Path(exists=True),
        help="Path to configuration file"
    )(f)
    f = click.option(
        "--env-file",
        type=click.Path(exists=True),
        help="Path to environment file"
    )(f)
    f = click.option(
        "--verbose", "-v",
        is_flag=True,
        help="Enable verbose logging"
    )(f)
    return f


def setup_cli_context(config_file: Optional[str], env_file: Optional[str], verbose: bool):
    """
    Setup CLI context with common options.

    Args:
        config_file: Path to configuration file
        env_file: Path to environment file
        verbose: Enable verbose logging
    """
    # Create and initialize context
    context = CLIContext(config_file, env_file)

    # Set verbose logging if requested
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return context