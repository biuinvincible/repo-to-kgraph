"""
Database connection and session management.

Provides connection pools and session management for Neo4j and ChromaDB
with health checking, reconnection logic, and resource cleanup.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager
import time

try:
    from neo4j import GraphDatabase, Driver, Session, AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError
except ImportError:
    GraphDatabase = None
    Driver = None
    Session = None
    AsyncGraphDatabase = None
    AsyncDriver = None
    AsyncSession = None

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb import Client as ChromaClient
except ImportError:
    chromadb = None
    ChromaClient = None

from repo_kgraph.lib.config import Config, DatabaseConfig


logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class Neo4jManager:
    """
    Neo4j database connection manager.

    Handles connection pooling, session management, and health checking
    for Neo4j graph database operations.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize Neo4j connection manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._driver: Optional[AsyncDriver] = None
        self._sync_driver: Optional[Driver] = None
        self._connection_attempts = 0
        self._max_connection_attempts = 3
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds

    async def initialize(self) -> None:
        """Initialize Neo4j connection."""
        if not AsyncGraphDatabase:
            raise DatabaseConnectionError("Neo4j driver not available")

        try:
            # Create async driver
            self._driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password),
                max_connection_lifetime=self.config.neo4j_max_connection_lifetime,
                max_connection_pool_size=self.config.neo4j_max_connection_pool_size
            )

            # Create sync driver for operations that need it
            if GraphDatabase:
                self._sync_driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_username, self.config.neo4j_password),
                    max_connection_lifetime=self.config.neo4j_max_connection_lifetime,
                    max_connection_pool_size=self.config.neo4j_max_connection_pool_size
                )

            # Verify connection
            await self.verify_connection()

            logger.info(f"Neo4j connection established: {self.config.neo4j_uri}")

        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection: {e}")
            raise DatabaseConnectionError(f"Neo4j connection failed: {e}")

    async def verify_connection(self) -> bool:
        """
        Verify Neo4j connection is working.

        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._driver:
            return False

        try:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                return record and record["test"] == 1

        except Exception as e:
            logger.warning(f"Neo4j connection verification failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dictionary with health status and metrics
        """
        current_time = time.time()

        # Skip if recent check
        if current_time - self._last_health_check < self._health_check_interval:
            return {"status": "cached", "healthy": self._driver is not None}

        self._last_health_check = current_time

        health_info = {
            "status": "unknown",
            "healthy": False,
            "connection_attempts": self._connection_attempts,
            "driver_available": self._driver is not None
        }

        try:
            if await self.verify_connection():
                health_info.update({
                    "status": "healthy",
                    "healthy": True,
                    "database": self.config.neo4j_database,
                    "uri": self.config.neo4j_uri
                })

                # Get additional database info
                try:
                    db_info = await self._get_database_info()
                    health_info.update(db_info)
                except Exception as e:
                    logger.warning(f"Failed to get database info: {e}")

            else:
                health_info["status"] = "unhealthy"

        except Exception as e:
            health_info.update({
                "status": "error",
                "error": str(e)
            })

        return health_info

    async def _get_database_info(self) -> Dict[str, Any]:
        """Get additional database information."""
        if not self._driver:
            return {}

        try:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                # Get node count
                result = await session.run("MATCH (n) RETURN count(n) as node_count")
                record = await result.single()
                node_count = record["node_count"] if record else 0

                # Get relationship count
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                record = await result.single()
                rel_count = record["rel_count"] if record else 0

                return {
                    "node_count": node_count,
                    "relationship_count": rel_count
                }

        except Exception as e:
            logger.warning(f"Failed to get database statistics: {e}")
            return {}

    @asynccontextmanager
    async def session(self, **kwargs) -> AsyncContextManager[AsyncSession]:
        """
        Get async Neo4j session as context manager.

        Args:
            **kwargs: Additional session parameters

        Yields:
            Neo4j async session
        """
        if not self._driver:
            raise DatabaseConnectionError("Neo4j driver not initialized")

        session_kwargs = {"database": self.config.neo4j_database}
        session_kwargs.update(kwargs)

        async with self._driver.session(**session_kwargs) as session:
            yield session

    def sync_session(self, **kwargs) -> Session:
        """
        Get synchronous Neo4j session.

        Args:
            **kwargs: Additional session parameters

        Returns:
            Neo4j session
        """
        if not self._sync_driver:
            raise DatabaseConnectionError("Neo4j sync driver not initialized")

        session_kwargs = {"database": self.config.neo4j_database}
        session_kwargs.update(kwargs)

        return self._sync_driver.session(**session_kwargs)

    async def close(self) -> None:
        """Close Neo4j connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None

        if self._sync_driver:
            self._sync_driver.close()
            self._sync_driver = None

        logger.info("Neo4j connections closed")

    @property
    def is_connected(self) -> bool:
        """Check if driver is available."""
        return self._driver is not None


class ChromaManager:
    """
    ChromaDB vector database connection manager.

    Handles connection to ChromaDB with health checking and
    collection management.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize ChromaDB connection manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._client: Optional[ChromaClient] = None
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds

    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        if not chromadb:
            raise DatabaseConnectionError("ChromaDB not available")

        try:
            # Create persistent client
            self._client = chromadb.PersistentClient(path=self.config.chroma_db_path)

            # Verify connection by listing collections
            await self.verify_connection()

            logger.info(f"ChromaDB connection established: {self.config.chroma_db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB connection: {e}")
            raise DatabaseConnectionError(f"ChromaDB connection failed: {e}")

    async def verify_connection(self) -> bool:
        """
        Verify ChromaDB connection is working.

        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._client:
            return False

        try:
            # Test connection by listing collections
            collections = self._client.list_collections()
            return True

        except Exception as e:
            logger.warning(f"ChromaDB connection verification failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dictionary with health status and metrics
        """
        current_time = time.time()

        # Skip if recent check
        if current_time - self._last_health_check < self._health_check_interval:
            return {"status": "cached", "healthy": self._client is not None}

        self._last_health_check = current_time

        health_info = {
            "status": "unknown",
            "healthy": False,
            "client_available": self._client is not None,
            "db_path": self.config.chroma_db_path
        }

        try:
            if await self.verify_connection():
                health_info.update({
                    "status": "healthy",
                    "healthy": True
                })

                # Get collection info
                try:
                    collection_info = await self._get_collection_info()
                    health_info.update(collection_info)
                except Exception as e:
                    logger.warning(f"Failed to get collection info: {e}")

            else:
                health_info["status"] = "unhealthy"

        except Exception as e:
            health_info.update({
                "status": "error",
                "error": str(e)
            })

        return health_info

    async def _get_collection_info(self) -> Dict[str, Any]:
        """Get ChromaDB collection information."""
        if not self._client:
            return {}

        try:
            collections = self._client.list_collections()
            collection_names = [col.name for col in collections]

            # Get repository collections
            repo_collections = [name for name in collection_names
                              if name.startswith(self.config.chroma_collection_prefix)]

            return {
                "total_collections": len(collections),
                "repository_collections": len(repo_collections),
                "collection_names": collection_names[:10]  # Limit for readability
            }

        except Exception as e:
            logger.warning(f"Failed to get collection statistics: {e}")
            return {}

    def get_client(self) -> ChromaClient:
        """
        Get ChromaDB client.

        Returns:
            ChromaDB client instance

        Raises:
            DatabaseConnectionError: If client not initialized
        """
        if not self._client:
            raise DatabaseConnectionError("ChromaDB client not initialized")

        return self._client

    def get_or_create_collection(self, name: str, **kwargs):
        """
        Get or create a ChromaDB collection.

        Args:
            name: Collection name
            **kwargs: Additional collection parameters

        Returns:
            ChromaDB collection
        """
        if not self._client:
            raise DatabaseConnectionError("ChromaDB client not initialized")

        return self._client.get_or_create_collection(name=name, **kwargs)

    def delete_collection(self, name: str) -> bool:
        """
        Delete a ChromaDB collection.

        Args:
            name: Collection name

        Returns:
            True if deletion successful, False otherwise
        """
        if not self._client:
            return False

        try:
            self._client.delete_collection(name=name)
            logger.info(f"Deleted ChromaDB collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False

    async def close(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB client doesn't need explicit closing
        self._client = None
        logger.info("ChromaDB connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if client is available."""
        return self._client is not None


class DatabaseManager:
    """
    Central database manager coordinating all database connections.

    Manages Neo4j and ChromaDB connections with unified health checking
    and resource management.
    """

    def __init__(self, config: Config):
        """
        Initialize database manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.neo4j = Neo4jManager(config.database)
        self.chroma = ChromaManager(config.database)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all database connections."""
        if self._initialized:
            return

        logger.info("Initializing database connections...")

        # Initialize Neo4j
        try:
            await self.neo4j.initialize()
        except Exception as e:
            logger.error(f"Neo4j initialization failed: {e}")
            # Don't fail completely if Neo4j is unavailable
            pass

        # Initialize ChromaDB
        try:
            await self.chroma.initialize()
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            # Don't fail completely if ChromaDB is unavailable
            pass

        self._initialized = True
        logger.info("Database initialization completed")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all database connections.

        Returns:
            Dictionary with overall health status
        """
        neo4j_health = await self.neo4j.health_check()
        chroma_health = await self.chroma.health_check()

        overall_healthy = neo4j_health.get("healthy", False) and chroma_health.get("healthy", False)

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "healthy": overall_healthy,
            "components": {
                "neo4j": neo4j_health,
                "chroma": chroma_health
            }
        }

    async def close(self) -> None:
        """Close all database connections."""
        await self.neo4j.close()
        await self.chroma.close()
        self._initialized = False
        logger.info("All database connections closed")

    @property
    def is_healthy(self) -> bool:
        """Check if all database connections are healthy."""
        return self.neo4j.is_connected and self.chroma.is_connected

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


async def get_database_manager(config: Optional[Config] = None) -> DatabaseManager:
    """
    Get global database manager instance.

    Args:
        config: Optional configuration (uses default if not provided)

    Returns:
        DatabaseManager instance
    """
    global _database_manager

    if _database_manager is None:
        if config is None:
            from repo_kgraph.lib.config import get_config
            config = get_config()

        _database_manager = DatabaseManager(config)
        await _database_manager.initialize()

    return _database_manager


async def close_database_connections() -> None:
    """Close global database connections."""
    global _database_manager

    if _database_manager:
        await _database_manager.close()
        _database_manager = None


# Connection utilities
@asynccontextmanager
async def neo4j_session(config: Optional[Config] = None, **kwargs):
    """
    Get Neo4j session as context manager.

    Args:
        config: Optional configuration
        **kwargs: Additional session parameters

    Yields:
        Neo4j async session
    """
    db_manager = await get_database_manager(config)
    async with db_manager.neo4j.session(**kwargs) as session:
        yield session


def get_chroma_client(config: Optional[Config] = None) -> ChromaClient:
    """
    Get ChromaDB client.

    Args:
        config: Optional configuration

    Returns:
        ChromaDB client
    """
    # Note: This is a simplified version for sync usage
    # In practice, you'd want to ensure the async initialization is complete
    if config is None:
        from repo_kgraph.lib.config import get_config
        config = get_config()

    if not chromadb:
        raise DatabaseConnectionError("ChromaDB not available")

    return chromadb.PersistentClient(path=config.database.chroma_db_path)