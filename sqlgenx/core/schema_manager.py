"""Enhanced schema management with dual format storage and sync."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from sqlgenx.core.schema_loader import SchemaInfo, TableInfo, ColumnInfo


@dataclass
class SemanticRelationship:
    """Semantic description of table relationships."""
    source_table: str
    target_table: str
    relationship_type: str  # "one-to-many", "many-to-many", "one-to-one"
    description: str  # Natural language description
    join_path: List[str]  # List of tables in join path


@dataclass
class SchemaMetadata:
    """Metadata about schema state."""
    schema_hash: str
    last_updated: str
    table_count: int
    column_count: int
    relationship_count: int
    dbms_type: str
    version: str = "1.0"


class SchemaManager:
    """Manages schema storage in dual format (JSON + SQL) with semantic enrichment."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.schema_json_path = workspace_dir / "schema.json"
        self.schema_sql_path = workspace_dir / "schema.sql"
        self.schema_index_path = workspace_dir / "schema_index.json"
        self.metadata_path = workspace_dir / "schema_metadata.json"
    
    def save_schema(
        self, 
        schema_info: SchemaInfo,
        semantic_relationships: Optional[List[SemanticRelationship]] = None
    ) -> str:
        """Save schema in both JSON and SQL formats with auto-sync."""
        
        # Convert to JSON-friendly format
        schema_dict = self._schema_to_dict(schema_info, semantic_relationships)
        
        # Compute hash for change detection
        schema_hash = self._compute_hash(schema_dict)
        
        # Save JSON (machine-readable)
        with open(self.schema_json_path, 'w') as f:
            json.dump(schema_dict, f, indent=2)
        
        # Save SQL (human-readable)
        sql_content = self._generate_sql_ddl(schema_info)
        with open(self.schema_sql_path, 'w') as f:
            f.write(sql_content)
        
        # Create index for fast lookup
        self._create_index(schema_dict)
        
        # Save metadata
        metadata = SchemaMetadata(
            schema_hash=schema_hash,
            last_updated=datetime.now().isoformat(),
            table_count=len(schema_info.tables),
            column_count=sum(len(t.columns) for t in schema_info.tables),
            relationship_count=len(semantic_relationships) if semantic_relationships else 0,
            dbms_type=schema_dict.get('dbms_type', 'generic')
        )
        self._save_metadata(metadata)
        
        return schema_hash
    
    def load_schema(self) -> Optional[Dict[str, Any]]:
        """Load schema from JSON (primary source)."""
        if not self.schema_json_path.exists():
            return None
        
        with open(self.schema_json_path, 'r') as f:
            return json.load(f)
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Fast lookup of table info using index."""
        index = self._load_index()
        if not index or table_name not in index.get('tables', {}):
            return None
        
        # Load from main schema
        schema = self.load_schema()
        if not schema:
            return None
        
        table_idx = index['tables'][table_name]['index']
        return schema['tables'][table_idx]
    
    def check_schema_changed(self, current_hash: str) -> bool:
        """Check if schema has changed since given hash."""
        metadata = self._load_metadata()
        if not metadata:
            return True
        
        return metadata.get('schema_hash') != current_hash
    
    def get_metadata(self) -> Optional[SchemaMetadata]:
        """Get schema metadata."""
        data = self._load_metadata()
        if not data:
            return None
        
        return SchemaMetadata(**data)
    
    def _schema_to_dict(
        self, 
        schema_info: SchemaInfo,
        semantic_relationships: Optional[List[SemanticRelationship]]
    ) -> Dict[str, Any]:
        """Convert SchemaInfo to dictionary with semantic enrichment."""
        
        tables_data = []
        for table in schema_info.tables:
            table_dict = {
                'name': table.name,
                'columns': [
                    {
                        'name': col.name,
                        'type': col.data_type,
                        'nullable': col.nullable,
                        'primary_key': col.primary_key,
                        'foreign_key': col.foreign_key,
                        'default': col.default
                    }
                    for col in table.columns
                ],
                'primary_keys': table.primary_keys,
                'foreign_keys': table.foreign_keys,
                'indexes': table.indexes
            }
            tables_data.append(table_dict)
        
        schema_dict = {
            'version': '1.0',
            'dbms_type': 'generic',  # Override from metadata if available
            'tables': tables_data,
            'semantic_relationships': [
                asdict(rel) for rel in semantic_relationships
            ] if semantic_relationships else []
        }
        
        return schema_dict
    
    def _generate_sql_ddl(self, schema_info: SchemaInfo) -> str:
        """Generate SQL DDL statements."""
        sql_parts = [
            "-- SQLGenX Schema Export",
            f"-- Generated: {datetime.now().isoformat()}",
            f"-- Tables: {len(schema_info.tables)}",
            ""
        ]
        
        for table in schema_info.tables:
            table_sql = [f"CREATE TABLE {table.name} ("]
            
            col_defs = []
            for col in table.columns:
                col_def = f"    {col.name} {col.data_type}"
                
                if col.primary_key:
                    col_def += " PRIMARY KEY"
                
                if not col.nullable and not col.primary_key:
                    col_def += " NOT NULL"
                
                if col.default:
                    col_def += f" DEFAULT {col.default}"
                
                col_defs.append(col_def)
            
            # Add foreign keys
            for col_name, ref in table.foreign_keys.items():
                fk_def = f"    FOREIGN KEY ({col_name}) REFERENCES {ref}"
                col_defs.append(fk_def)
            
            table_sql.append(",\n".join(col_defs))
            table_sql.append(");")
            table_sql.append("")
            
            sql_parts.append("\n".join(table_sql))
        
        return "\n".join(sql_parts)
    
    def _create_index(self, schema_dict: Dict[str, Any]) -> None:
        """Create fast lookup index."""
        index = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'tables': {},
            'columns': {}
        }
        
        # Index tables
        for idx, table in enumerate(schema_dict['tables']):
            table_name = table['name']
            index['tables'][table_name] = {
                'index': idx,
                'column_count': len(table['columns']),
                'has_fk': len(table.get('foreign_keys', {})) > 0
            }
            
            # Index columns for search
            for col in table['columns']:
                col_name = col['name']
                if col_name not in index['columns']:
                    index['columns'][col_name] = []
                
                index['columns'][col_name].append({
                    'table': table_name,
                    'type': col['type'],
                    'primary_key': col.get('primary_key', False)
                })
        
        with open(self.schema_index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _load_index(self) -> Optional[Dict[str, Any]]:
        """Load schema index."""
        if not self.schema_index_path.exists():
            return None
        
        with open(self.schema_index_path, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: SchemaMetadata) -> None:
        """Save schema metadata."""
        with open(self.metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load schema metadata."""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def _compute_hash(self, schema_dict: Dict[str, Any]) -> str:
        """Compute hash of schema structure."""
        # Use canonical JSON representation
        canonical = json.dumps(
            schema_dict, 
            sort_keys=True, 
            default=str
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class SemanticEnricher:
    """Enrich schema with semantic relationships and natural language descriptions."""
    
    @staticmethod
    def detect_relationships(schema_info: SchemaInfo) -> List[SemanticRelationship]:
        """Detect and describe relationships between tables."""
        relationships = []
        
        # Build table map
        table_map = {t.name: t for t in schema_info.tables}
        
        for table in schema_info.tables:
            # Detect foreign key relationships
            for col_name, ref in table.foreign_keys.items():
                if '.' in ref:
                    ref_table, ref_col = ref.split('.')
                    
                    # Determine relationship type
                    rel_type = "many-to-one"  # Default
                    
                    # Check if it's one-to-one (FK is also unique)
                    col_info = next((c for c in table.columns if c.name == col_name), None)
                    if col_info and col_info.primary_key:
                        rel_type = "one-to-one"
                    
                    description = f"{table.name} has {rel_type.replace('-', ' to ')} relationship with {ref_table}"
                    
                    relationships.append(SemanticRelationship(
                        source_table=table.name,
                        target_table=ref_table,
                        relationship_type=rel_type,
                        description=description,
                        join_path=[table.name, ref_table]
                    ))
        
        # Detect many-to-many through junction tables
        relationships.extend(
            SemanticEnricher._detect_many_to_many(schema_info, table_map)
        )
        
        return relationships
    
    @staticmethod
    def _detect_many_to_many(
        schema_info: SchemaInfo, 
        table_map: Dict[str, TableInfo]
    ) -> List[SemanticRelationship]:
        """Detect many-to-many relationships via junction tables."""
        m2m_relationships = []
        
        for table in schema_info.tables:
            # Junction table heuristic: has exactly 2 foreign keys, small name
            if len(table.foreign_keys) == 2:
                fk_refs = list(table.foreign_keys.values())
                
                if len(fk_refs) == 2:
                    ref1 = fk_refs[0].split('.')[0]
                    ref2 = fk_refs[1].split('.')[0]
                    
                    description = (
                        f"{ref1} are related to {ref2} through {table.name} "
                        f"(many-to-many relationship)"
                    )
                    
                    m2m_relationships.append(SemanticRelationship(
                        source_table=ref1,
                        target_table=ref2,
                        relationship_type="many-to-many",
                        description=description,
                        join_path=[ref1, table.name, ref2]
                    ))
        
        return m2m_relationships
    
    @staticmethod
    def generate_natural_language_schema(
        schema_dict: Dict[str, Any]
    ) -> str:
        """Generate natural language description of schema for LLM."""
        lines = [
            "# Database Schema Overview",
            "",
            f"**Total Tables:** {len(schema_dict['tables'])}",
            ""
        ]
        
        # Add semantic relationships first
        if schema_dict.get('semantic_relationships'):
            lines.append("## Relationships")
            lines.append("")
            for rel in schema_dict['semantic_relationships']:
                lines.append(f"- {rel['description']}")
            lines.append("")
        
        # Add table details
        lines.append("## Tables")
        lines.append("")
        
        for table in schema_dict['tables']:
            lines.append(f"### Table: `{table['name']}`")
            lines.append("")
            lines.append("**Columns:**")
            
            for col in table['columns']:
                col_desc = f"- `{col['name']}` ({col['type']})"
                
                attrs = []
                if col.get('primary_key'):
                    attrs.append("PRIMARY KEY")
                if not col.get('nullable'):
                    attrs.append("NOT NULL")
                if col.get('foreign_key'):
                    attrs.append("FOREIGN KEY")
                
                if attrs:
                    col_desc += f" — {', '.join(attrs)}"
                
                lines.append(col_desc)
            
            # Add foreign key details
            if table.get('foreign_keys'):
                lines.append("")
                lines.append("**References:**")
                for col, ref in table['foreign_keys'].items():
                    lines.append(f"- `{col}` → `{ref}`")
            
            lines.append("")
        
        return "\n".join(lines)