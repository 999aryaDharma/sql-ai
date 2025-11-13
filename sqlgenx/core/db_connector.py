"""Database connection manager for live database integration."""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd


class DatabaseConnection:
    """Manages live database connections."""
    
    def __init__(self, connection_string: str):
        """Initialize database connection."""
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None
        self.dialect: str = ""
        self._connect()
    
    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self.engine = create_engine(self.connection_string)
            self.dialect = self.engine.dialect.name
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test if connection is alive."""
        try:
            if not self.engine:
                return False
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Extract schema information from live database."""
        if not self.engine:
            raise ConnectionError("Not connected to database")
        
        inspector = inspect(self.engine)
        schema_info = {
            "tables": [],
            "dialect": self.dialect
        }
        
        for table_name in inspector.get_table_names():
            table_info = self._get_table_info(inspector, table_name)
            schema_info["tables"].append(table_info)
        
        return schema_info
    
    def _get_table_info(self, inspector, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a table."""
        columns = []
        
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"],
                "default": str(col.get("default", "")) if col.get("default") else None,
                "primary_key": False  # Will be updated below
            })
        
        # Get primary keys
        pk_constraint = inspector.get_pk_constraint(table_name)
        pk_columns = pk_constraint.get("constrained_columns", []) if pk_constraint else []
        
        for col in columns:
            if col["name"] in pk_columns:
                col["primary_key"] = True
        
        # Get foreign keys
        foreign_keys = {}
        try:
            for fk in inspector.get_foreign_keys(table_name):
                for col in fk.get("constrained_columns", []):
                    ref_table = fk.get("referred_table", "")
                    ref_cols = fk.get("referred_columns", [])
                    if ref_cols:
                        foreign_keys[col] = f"{ref_table}.{ref_cols[0]}"
        except Exception:
            # Some databases might not support foreign keys
            pass
        
        # Get indexes
        indexes = []
        try:
            indexes = [idx["name"] for idx in inspector.get_indexes(table_name) if idx.get("name")]
        except Exception:
            # Some databases might not support indexes
            pass
        
        return {
            "name": table_name,
            "columns": columns,
            "primary_keys": pk_columns,
            "foreign_keys": foreign_keys,
            "indexes": indexes,
            "column_samples": []  # Add empty list for column samples
        }
    
    def execute_query(
        self, 
        query: str, 
        limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute SQL query and return results as DataFrame."""
        if not self.engine:
            raise ConnectionError("Not connected to database")
        
        try:
            # Add LIMIT if specified and not already in query
            if limit and "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                # Get column names
                columns = list(result.keys()) if result.keys() else []
                
                # Fetch data
                rows = result.fetchall()
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)
                
                # Get execution metadata
                metadata = {
                    "row_count": len(df),
                    "column_count": len(columns),
                    "columns": columns,
                    "query": query
                }
                
                return df, metadata
        
        except SQLAlchemyError as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def get_table_preview(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Get a preview of table data."""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df, _ = self.execute_query(query)
        return df
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics about a table."""
        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            df, _ = self.execute_query(count_query)
            row_count = df.iloc[0]['count'] if not df.empty else 0
            
            # Get column info
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            return {
                "table_name": table_name,
                "row_count": int(row_count),
                "column_count": len(columns),
                "columns": [col["name"] for col in columns]
            }
        except Exception as e:
            return {
                "table_name": table_name,
                "error": str(e)
            }
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None


class ConnectionManager:
    """Manages saved database connections."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.connections_file = workspace_dir / "connections.json"
    
    def save_connection(
        self, 
        name: str, 
        connection_string: str,
        description: Optional[str] = None
    ) -> None:
        """Save a database connection."""
        connections = self._load_connections()
        
        connections[name] = {
            "connection_string": connection_string,
            "description": description,
            "dialect": self._get_dialect_from_string(connection_string)
        }
        
        self._save_connections(connections)
    
    def get_connection(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a saved connection."""
        connections = self._load_connections()
        return connections.get(name)
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """List all saved connections."""
        connections = self._load_connections()
        return [
            {"name": name, **info}
            for name, info in connections.items()
        ]
    
    def delete_connection(self, name: str) -> None:
        """Delete a saved connection."""
        connections = self._load_connections()
        if name in connections:
            del connections[name]
            self._save_connections(connections)
    
    def _load_connections(self) -> Dict[str, Any]:
        """Load connections from file."""
        if not self.connections_file.exists():
            return {}
        
        try:
            with open(self.connections_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    
    def _save_connections(self, connections: Dict[str, Any]) -> None:
        """Save connections to file."""
        with open(self.connections_file, "w") as f:
            json.dump(connections, f, indent=2)
    
    def _get_dialect_from_string(self, connection_string: str) -> str:
        """Extract dialect from connection string."""
        if connection_string.startswith("postgresql"):
            return "postgresql"
        elif connection_string.startswith("mysql"):
            return "mysql"
        elif connection_string.startswith("sqlite"):
            return "sqlite"
        else:
            return "unknown"


def build_connection_string(
    dialect: str,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
    **kwargs
) -> str:
    """Build a connection string from components."""
    if dialect == "postgresql":
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    elif dialect == "mysql":
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif dialect == "sqlite":
        return f"sqlite:///{database}"
    else:
        raise ValueError(f"Unsupported dialect: {dialect}")