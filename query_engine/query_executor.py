import logging
from typing import Optional, Dict, Any, Union, List
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import NullPool
import time
import asyncio

try:
    import asyncpg
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
except ImportError:
    asyncpg = None
    create_async_engine = None
    AsyncEngine = None
    AsyncSession = None

class QueryExecutor:
    """
    Securely executes SQL queries against relational databases with support for
    PostgreSQL, MySQL, and SQLite using SQLAlchemy.

    Features:
    - Parameterized execution
    - Engine pooling and multiple backends
    - Query timeouts, retries
    - Logging
    - Returns results as pandas DataFrames
    - Async execution for high concurrency
    """

    def __init__(
        self,
        db_url: str,
        logger: Optional[logging.Logger] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        async_mode: bool = False,
        connect_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            db_url: SQLAlchemy connection string.
            logger: Logger instance.
            pool_size: Number of connections in pool.
            max_overflow: Extra connections above pool_size.
            async_mode: Use async SQLAlchemy engine.
            connect_args: Optional connection arguments.
        """
        self.db_url = db_url
        self.logger = logger or logging.getLogger(__name__)
        self.async_mode = async_mode
        self.connect_args = connect_args or {}
        self.engine = None
        self.async_engine = None
        self._setup_engine(pool_size, max_overflow)

    def _setup_engine(self, pool_size: int, max_overflow: int):
        """
        Initialize SQLAlchemy engine (sync or async).
        """
        if self.async_mode:
            if create_async_engine is None:
                raise ImportError("asyncpg and sqlalchemy[asyncio] are required for async_mode")
            self.async_engine: AsyncEngine = create_async_engine(
                self.db_url,
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                connect_args=self.connect_args
            )
            self.logger.info("Initialized async SQLAlchemy engine.")
        else:
            self.engine: Engine = create_engine(
                self.db_url,
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                connect_args=self.connect_args
            )
            self.logger.info("Initialized sync SQLAlchemy engine.")

    def execute(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        timeout: Optional[float] = 30,
        retries: int = 2,
        retry_delay: float = 2.0,
    ) -> pd.DataFrame:
        """
        Execute a parameterized query safely and return results as a DataFrame.

        Args:
            query: SQL query (use :param or %s placeholders).
            params: Parameters to bind.
            timeout: Query timeout in seconds.
            retries: Number of times to retry on failure.
            retry_delay: Delay between retries.

        Returns:
            Pandas DataFrame with results.

        Raises:
            Exception with clear message on failure.
        """
        attempt = 0
        last_exception = None
        while attempt <= retries:
            try:
                self.logger.info(f"Executing query (attempt {attempt+1}): {query} | params: {params}")
                with self.engine.connect() as connection:
                    # Set timeout (works for PostgreSQL, MySQL; SQLite ignored)
                    if timeout and "postgresql" in self.db_url:
                        connection.execute(text(f"SET statement_timeout = {int(timeout * 1000)}"))
                    elif timeout and "mysql" in self.db_url:
                        connection.execute(text(f"SET SESSION MAX_EXECUTION_TIME={int(timeout * 1000)}"))
                    result = connection.execute(text(query), params or {})
                    if result.returns_rows:
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    else:
                        df = pd.DataFrame()
                self.logger.info("Query executed successfully.")
                return df
            except (OperationalError, ProgrammingError, SQLAlchemyError) as e:
                self.logger.error(f"Query failed: {e}")
                last_exception = e
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(f"Query failed after {retries+1} attempts: {e}")
                time.sleep(retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise RuntimeError(f"Unexpected error during query execution: {e}")

    async def aexecute(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        timeout: Optional[float] = 30,
        retries: int = 2,
        retry_delay: float = 2.0,
    ) -> pd.DataFrame:
        """
        Asynchronously execute a parameterized query and return a DataFrame.

        Args:
            query: SQL query (use :param or %s placeholders).
            params: Parameters to bind.
            timeout: Query timeout in seconds.
            retries: Number of times to retry on failure.
            retry_delay: Delay between retries.

        Returns:
            Pandas DataFrame with results.

        Raises:
            Exception with clear message on failure.
        """
        if not self.async_mode or self.async_engine is None:
            raise RuntimeError("Async mode is not enabled or engine not initialized.")
        attempt = 0
        last_exception = None
        while attempt <= retries:
            try:
                self.logger.info(f"Async executing query (attempt {attempt+1}): {query} | params: {params}")
                async with self.async_engine.connect() as conn:
                    if timeout and "postgresql" in self.db_url:
                        await conn.execute(text(f"SET statement_timeout = {int(timeout * 1000)}"))
                    elif timeout and "mysql" in self.db_url:
                        await conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={int(timeout * 1000)}"))
                    result = await conn.execute(text(query), params or {})
                    if result.returns_rows:
                        rows = await result.fetchall()
                        df = pd.DataFrame(rows, columns=result.keys())
                    else:
                        df = pd.DataFrame()
                self.logger.info("Async query executed successfully.")
                return df
            except (OperationalError, ProgrammingError, SQLAlchemyError) as e:
                self.logger.error(f"Async query failed: {e}")
                last_exception = e
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(f"Async query failed after {retries+1} attempts: {e}")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected async error: {e}")
                raise RuntimeError(f"Unexpected error during async query execution: {e}")

    def close(self):
        """
        Dispose of the SQLAlchemy engine.
        """
        if self.async_mode and self.async_engine is not None:
            self.logger.info("Disposing async engine.")
            asyncio.create_task(self.async_engine.dispose())
        elif self.engine is not None:
            self.logger.info("Disposing sync engine.")
            self.engine.dispose()

# Example usage:
# logger = logging.getLogger("query_executor")
# executor = QueryExecutor("postgresql://user:pass@host/dbname", logger=logger)
# df = executor.execute("SELECT * FROM users WHERE id = :id", params={"id": 5})

# For async:
# executor = QueryExecutor("postgresql+asyncpg://user:pass@host/dbname", async_mode=True)
# df = await executor.aexecute("SELECT * FROM users WHERE id = :id", params={"id": 5})
