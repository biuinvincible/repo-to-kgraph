"""
Database connection module with clear structure.

This module contains exactly 3 entities:
1. DatabaseConnection class
2. connect_to_database function
3. execute_query function

Expected relationships:
- connect_to_database RETURNS DatabaseConnection
- execute_query CALLS DatabaseConnection.execute (implicit)
"""

from user import User, validate_email  # IMPORT relationship


class DatabaseConnection:
    """Simple database connection wrapper."""

    def __init__(self, host: str, port: int, database: str):
        """Initialize database connection."""
        self.host = host
        self.port = port
        self.database = database
        self.connected = False

    def connect(self):
        """Establish database connection."""
        # Mock connection logic
        self.connected = True
        return True

    def execute(self, query: str):
        """Execute a database query."""
        if not self.connected:
            raise RuntimeError("Not connected to database")

        # Mock query execution
        return f"Executed: {query}"

    def close(self):
        """Close the database connection."""
        self.connected = False


def connect_to_database(host: str = "localhost", port: int = 5432) -> DatabaseConnection:
    """Create and return a database connection."""
    db_name = "test_db"
    connection = DatabaseConnection(host, port, db_name)
    connection.connect()
    return connection


def execute_query(query: str, connection: DatabaseConnection):
    """Execute a query using the provided connection."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Validate email in queries (uses imported function)
    if "email" in query.lower():
        # This creates a CALLS relationship to validate_email
        dummy_email = "test@example.com"
        validate_email(dummy_email)

    return connection.execute(query)