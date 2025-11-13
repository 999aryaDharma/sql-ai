"""Schema loader and parser using SQLGlot."""
from pathlib import Path
from typing import List, Dict, Any
import sqlglot
from sqlglot import exp
from pydantic import BaseModel


class ColumnInfo(BaseModel):
    """Information about a database column."""
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: bool = False
    default: str | None = None
    
    class Config:
        extra = "ignore"  # Ignore extra fields


class TableInfo(BaseModel):
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo]
    primary_keys: List[str] = []
    foreign_keys: Dict[str, str] = {}  # column -> referenced_table.column
    indexes: List[str] = []
    column_samples: List[str] = []  # Optional: sample values for context
    
    class Config:
        extra = "ignore"  # Ignore extra fields
    
    def to_text(self) -> str:
        """Convert table info to text representation for embedding."""
        lines = [f"Table: {self.name}"]
        lines.append("Columns:")
        
        for col in self.columns:
            pk = " PRIMARY KEY" if col.primary_key else ""
            fk = f" REFERENCES {self.foreign_keys.get(col.name, '')}" if col.foreign_key else ""
            nullable = "" if col.nullable else " NOT NULL"
            lines.append(f"  - {col.name} ({col.data_type}){pk}{fk}{nullable}")
        
        # Add summary for better vector search matches
        column_names = [col.name for col in self.columns]
        lines.append(f"Column names: {', '.join(column_names)}")
        
        if self.indexes:
            lines.append(f"Indexes: {', '.join(self.indexes)}")
        
        return "\n".join(lines)


class SchemaInfo(BaseModel):
    """Complete schema information."""
    tables: List[TableInfo]
    raw_sql: str
    
    def to_text_chunks(self) -> List[Dict[str, str]]:
        """Convert schema to text chunks for vector embedding."""
        chunks = []
        
        for table in self.tables:
            chunks.append({
                "table": table.name,
                "content": table.to_text(),
                "type": "table_definition"
            })
        
        return chunks


class SchemaLoader:
    """Load and parse SQL schema files."""
    
    def __init__(self, dialect: str = ""):
        """Initialize schema loader with optional dialect."""
        self.dialect = dialect or None
    
    def load_from_file(self, file_path: Path) -> SchemaInfo:
        """Load schema from a SQL file."""
        with open(file_path, "r", encoding="utf-8") as f:
            sql_content = f.read()
        
        return self.parse_schema(sql_content)
    
    def parse_schema(self, sql_content: str) -> SchemaInfo:
        """Parse SQL schema content."""
        tables = []
        
        try:
            # Parse SQL statements
            statements = sqlglot.parse(sql_content, read=self.dialect)
            
            for statement in statements:
                if isinstance(statement, exp.Create):
                    table_info = self._parse_create_table(statement)
                    if table_info:
                        tables.append(table_info)
        
        except Exception as e:
            # Fallback to basic parsing if SQLGlot fails
            print(f"Warning: SQLGlot parsing failed: {e}")
            tables = self._fallback_parse(sql_content)
        
        return SchemaInfo(tables=tables, raw_sql=sql_content)
    
    def _parse_create_table(self, statement: exp.Create) -> TableInfo | None:
        """Parse a CREATE TABLE statement."""
        if not statement.this:
            return None
        
        table_name = statement.this.this.name
        columns = []
        primary_keys = []
        foreign_keys = {}
        indexes = []
        
        # Parse columns
        if hasattr(statement, 'this') and hasattr(statement.this, 'expressions'):
            for expr in statement.this.expressions:
                if isinstance(expr, exp.ColumnDef):
                    col_info = self._parse_column(expr)
                    columns.append(col_info)
                    
                    if col_info.primary_key:
                        primary_keys.append(col_info.name)
                    
                    # Check for inline foreign key
                    for constraint in expr.constraints:
                        if isinstance(constraint, exp.ForeignKey):
                            ref_table = constraint.reference.this.name if constraint.reference else ""
                            ref_col = constraint.reference.expressions[0].name if constraint.reference and constraint.reference.expressions else ""
                            foreign_keys[col_info.name] = f"{ref_table}.{ref_col}"
                            col_info.foreign_key = True
        
        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            indexes=indexes
        )
    
    def _parse_column(self, column_def: exp.ColumnDef) -> ColumnInfo:
        """Parse a column definition."""
        name = column_def.this.name
        data_type = str(column_def.kind) if column_def.kind else "UNKNOWN"
        
        nullable = True
        primary_key = False
        default = None
        
        # Check constraints
        for constraint in column_def.constraints:
            if isinstance(constraint, exp.NotNullColumnConstraint):
                nullable = False
            elif isinstance(constraint, exp.PrimaryKeyColumnConstraint):
                primary_key = True
                nullable = False
            elif isinstance(constraint, exp.DefaultColumnConstraint):
                default = str(constraint.this) if constraint.this else None
        
        return ColumnInfo(
            name=name,
            data_type=data_type,
            nullable=nullable,
            primary_key=primary_key,
            foreign_key=False,
            default=default
        )
    
    def _fallback_parse(self, sql_content: str) -> List[TableInfo]:
        """Fallback parser for when SQLGlot fails."""
        # Basic regex-based parsing as fallback
        tables = []
        lines = sql_content.split("\n")
        
        # This is a very basic implementation
        # In production, you might want a more robust fallback
        
        return tables