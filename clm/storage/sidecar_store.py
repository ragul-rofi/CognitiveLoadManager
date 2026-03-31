"""SQLite-based storage backend for compressed task chunks."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from clm.core.models import TaskChunk
from clm.exceptions import StorageError

logger = logging.getLogger("clm.sidecar_store")


class SidecarStore:
    """
    Persistent storage backend for compressed task chunks.
    Uses SQLite for local storage with optional Redis support (future).
    """
    
    def __init__(self, storage_type: str = "sqlite", connection_params: Optional[dict] = None):
        """
        Initialize storage backend.
        
        Args:
            storage_type: "sqlite" or "redis" (only sqlite supported in v1.0)
            connection_params: Connection parameters for storage backend
                             For SQLite: {"db_path": "path/to/db.sqlite"}
                             If not provided, uses in-memory database
                             
        Raises:
            StorageError: If storage initialization fails
        """
        if storage_type != "sqlite":
            error_msg = f"Only 'sqlite' storage type is supported, got '{storage_type}'"
            logger.error(error_msg)
            raise StorageError(error_msg)
        
        self.storage_type = storage_type
        self.connection_params = connection_params or {}
        
        # Initialize SQLite connection
        db_path = self.connection_params.get("db_path", ":memory:")
        self.db_path = db_path
        
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Initialized SQLite storage at {db_path}")
            
            # Create schema
            self._create_schema()
            
        except Exception as e:
            error_msg = f"Failed to initialize SQLite storage at {db_path}: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e
    
    def _create_schema(self) -> None:
        """Create SQLite schema with task_chunks table and indexes."""
        cursor = self.conn.cursor()
        
        # Create task_chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_chunks (
                task_id TEXT PRIMARY KEY,
                parent_id TEXT,
                summary TEXT NOT NULL,
                full_detail TEXT NOT NULL,
                clm_score_at_compression REAL NOT NULL,
                compressed_at TIMESTAMP NOT NULL,
                status TEXT NOT NULL
            )
        """)
        
        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_id 
            ON task_chunks(parent_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON task_chunks(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_compressed_at 
            ON task_chunks(compressed_at)
        """)
        
        self.conn.commit()

    def store(self, task_chunk: TaskChunk) -> str:
        """
        Persist task chunk with all metadata.
        
        Args:
            task_chunk: Contains task_id, parent_id, summary, full_detail,
                       clm_score_at_compression, compressed_at, status
                       
        Returns:
            task_id of stored chunk
            
        Raises:
            StorageError: If storage operation fails
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert datetime to ISO format string for storage
            compressed_at_str = task_chunk.compressed_at.isoformat()
            
            # Use INSERT OR REPLACE to handle updates
            cursor.execute("""
                INSERT OR REPLACE INTO task_chunks 
                (task_id, parent_id, summary, full_detail, clm_score_at_compression, 
                 compressed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_chunk.task_id,
                task_chunk.parent_id,
                task_chunk.summary,
                task_chunk.full_detail,
                task_chunk.clm_score_at_compression,
                compressed_at_str,
                task_chunk.status
            ))
            
            self.conn.commit()
            logger.debug(f"Stored task chunk {task_chunk.task_id} with status={task_chunk.status}")
            return task_chunk.task_id
            
        except Exception as e:
            error_msg = f"Failed to store task chunk {task_chunk.task_id}: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e
    
    def get(self, task_id: str) -> Optional[TaskChunk]:
        """
        Retrieve task chunk by ID.
        
        Args:
            task_id: ID of task chunk to retrieve
            
        Returns:
            TaskChunk if found, None otherwise
            
        Raises:
            StorageError: If retrieval operation fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT task_id, parent_id, summary, full_detail, 
                       clm_score_at_compression, compressed_at, status
                FROM task_chunks
                WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if row is None:
                logger.debug(f"Task chunk {task_id} not found")
                return None
            
            # Convert row to TaskChunk
            task_chunk = TaskChunk(
                task_id=row["task_id"],
                parent_id=row["parent_id"],
                summary=row["summary"],
                full_detail=row["full_detail"],
                clm_score_at_compression=row["clm_score_at_compression"],
                compressed_at=datetime.fromisoformat(row["compressed_at"]),
                status=row["status"]
            )
            logger.debug(f"Retrieved task chunk {task_id}")
            return task_chunk
            
        except Exception as e:
            error_msg = f"Failed to retrieve task chunk {task_id}: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e
    
    def list_children(self, parent_id: str) -> list[TaskChunk]:
        """
        Retrieve all child task chunks for given parent.
        
        Args:
            parent_id: ID of parent task
            
        Returns:
            List of TaskChunk objects with matching parent_id
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT task_id, parent_id, summary, full_detail, 
                   clm_score_at_compression, compressed_at, status
            FROM task_chunks
            WHERE parent_id = ?
            ORDER BY compressed_at DESC
        """, (parent_id,))
        
        rows = cursor.fetchall()
        
        # Convert rows to TaskChunk objects
        children = []
        for row in rows:
            children.append(TaskChunk(
                task_id=row["task_id"],
                parent_id=row["parent_id"],
                summary=row["summary"],
                full_detail=row["full_detail"],
                clm_score_at_compression=row["clm_score_at_compression"],
                compressed_at=datetime.fromisoformat(row["compressed_at"]),
                status=row["status"]
            ))
        
        return children
    
    def expand(self, task_id: str) -> Optional[str]:
        """
        Retrieve full_detail field for expansion.
        
        Args:
            task_id: ID of task to expand
            
        Returns:
            full_detail string if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT full_detail
            FROM task_chunks
            WHERE task_id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        return row["full_detail"]
    
    def get_stats(self) -> dict:
        """
        Return storage statistics.
        
        Returns:
            Dictionary with count, total_size, compressed_count, expanded_count
        """
        cursor = self.conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as count FROM task_chunks")
        total_count = cursor.fetchone()["count"]
        
        # Get count by status
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM task_chunks 
            GROUP BY status
        """)
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Get total size (sum of full_detail lengths)
        cursor.execute("""
            SELECT SUM(LENGTH(full_detail)) as total_size 
            FROM task_chunks
        """)
        total_size = cursor.fetchone()["total_size"] or 0
        
        return {
            "count": total_count,
            "total_size": total_size,
            "compressed_count": status_counts.get("compressed", 0),
            "expanded_count": status_counts.get("expanded", 0),
            "db_path": self.db_path
        }
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
