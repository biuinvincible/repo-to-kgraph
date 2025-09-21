"""
Database utility for SQLite operations.
Provides async database connectivity and query execution.
"""

import aiosqlite
import asyncio
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager


class Database:
    """Async database manager for SQLite operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = None

    async def connect(self) -> None:
        """Establish database connection."""
        try:
            self.connection = await aiosqlite.connect(self.database_url)
            self.connection.row_factory = aiosqlite.Row
            await self.connection.execute("PRAGMA foreign_keys = ON")
            await self.connection.commit()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None

    async def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        return_id: bool = False
    ) -> Union[int, Any]:
        """Execute a query and return affected rows or last insert id."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            cursor = await self.connection.execute(query, params or ())

            if return_id:
                result = cursor.lastrowid
            else:
                result = cursor.rowcount

            await self.connection.commit()
            return result

        except Exception as e:
            await self.connection.rollback()
            raise DatabaseError(f"Query execution failed: {e}")

    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from the database."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            cursor = await self.connection.execute(query, params or ())
            row = await cursor.fetchone()

            if row:
                return dict(row)
            return None

        except Exception as e:
            raise DatabaseError(f"Fetch one failed: {e}")

    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from the database."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            cursor = await self.connection.execute(query, params or ())
            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            raise DatabaseError(f"Fetch all failed: {e}")

    async def fetch_many(
        self,
        query: str,
        params: Optional[tuple] = None,
        size: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch a limited number of rows from the database."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            cursor = await self.connection.execute(query, params or ())
            rows = await cursor.fetchmany(size)

            return [dict(row) for row in rows]

        except Exception as e:
            raise DatabaseError(f"Fetch many failed: {e}")

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            await self.connection.execute("BEGIN")
            yield
            await self.connection.commit()
        except Exception as e:
            await self.connection.rollback()
            raise DatabaseError(f"Transaction failed: {e}")

    async def execute_script(self, script: str) -> None:
        """Execute a multi-statement SQL script."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            await self.connection.executescript(script)
            await self.connection.commit()
        except Exception as e:
            await self.connection.rollback()
            raise DatabaseError(f"Script execution failed: {e}")

    async def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get information about a table structure."""
        query = f"PRAGMA table_info({table_name})"
        return await self.fetch_all(query)

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        result = await self.fetch_one(query, (table_name,))
        return result is not None

    async def get_database_size(self) -> int:
        """Get the size of the database in bytes."""
        query = "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
        result = await self.fetch_one(query)
        return result['size'] if result else 0

    async def vacuum(self) -> None:
        """Optimize database by reclaiming unused space."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            await self.connection.execute("VACUUM")
            await self.connection.commit()
        except Exception as e:
            raise DatabaseError(f"Vacuum failed: {e}")

    async def backup(self, backup_path: str) -> None:
        """Create a backup of the database."""
        if not self.connection:
            raise ConnectionError("Database not connected")

        try:
            backup_connection = await aiosqlite.connect(backup_path)
            await self.connection.backup(backup_connection)
            await backup_connection.close()
        except Exception as e:
            raise DatabaseError(f"Backup failed: {e}")

    async def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current database connection."""
        if not self.connection:
            return {"connected": False}

        try:
            # Get database version
            version_result = await self.fetch_one("SELECT sqlite_version() as version")

            # Get database file size
            size = await self.get_database_size()

            # Get number of tables
            tables_result = await self.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )

            return {
                "connected": True,
                "database_url": self.database_url,
                "sqlite_version": version_result['version'] if version_result else "unknown",
                "database_size_bytes": size,
                "table_count": len(tables_result)
            }
        except Exception as e:
            return {
                "connected": True,
                "error": str(e)
            }


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class ConnectionPool:
    """Simple connection pool for SQLite databases."""

    def __init__(self, database_url: str, max_connections: int = 5):
        self.database_url = database_url
        self.max_connections = max_connections
        self.pool = asyncio.Queue(maxsize=max_connections)
        self.created_connections = 0

    async def get_connection(self) -> Database:
        """Get a database connection from the pool."""
        try:
            # Try to get an existing connection
            db = self.pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create a new connection if under limit
            if self.created_connections < self.max_connections:
                db = Database(self.database_url)
                await db.connect()
                self.created_connections += 1
            else:
                # Wait for an available connection
                db = await self.pool.get()

        return db

    async def return_connection(self, db: Database) -> None:
        """Return a connection to the pool."""
        try:
            self.pool.put_nowait(db)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            await db.disconnect()
            self.created_connections -= 1

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                db = self.pool.get_nowait()
                await db.disconnect()
            except asyncio.QueueEmpty:
                break

        self.created_connections = 0