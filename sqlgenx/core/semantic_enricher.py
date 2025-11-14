"""
sqlgenx/core/semantic_enricher.py

Semantic Enrichment Pipeline for SQLGenX
Transforms structural schema into rich semantic documents for better retrieval.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import re
from datetime import datetime
from sqlgenx.core.schema_loader import SchemaInfo, TableInfo, ColumnInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FieldSemantics:
    """Semantic metadata for a field."""
    column_name: str
    business_meaning: str
    synonyms: List[str]
    data_category: str  # temporal, financial, identifier, text, boolean, etc.
    is_metric: bool
    metric_type: Optional[str] = None  # sum, count, avg, ratio
    units: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class DerivedField:
    """Definition of a computed/derived field."""
    name: str
    expression: str
    description: str
    required_columns: List[str]
    output_type: str
    confidence: str = "high"  # high, medium, low


@dataclass
class JoinPath:
    """Describes how to join tables."""
    from_table: str
    to_table: str
    join_type: str  # one-to-many, many-to-one, many-to-many
    via_columns: List[Tuple[str, str]]  # [(left_col, right_col)]
    intermediate_tables: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TableSemantics:
    """Rich semantic metadata for a table."""
    table_name: str
    table_type: str  # fact, dimension, bridge, reference
    business_domain: str  # sales, inventory, auth, etc.
    business_purpose: str
    use_cases: List[str]
    field_semantics: List[FieldSemantics]
    derived_fields: List[DerivedField]
    common_metrics: List[str]
    related_tables: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticSchema:
    """Complete semantic enrichment of database schema."""
    schema_name: str
    version: str
    created_at: str
    structural_metadata: Dict[str, Any]
    table_semantics: List[TableSemantics]
    join_paths: List[JoinPath]
    domain_groupings: Dict[str, List[str]]  # domain -> [tables]
    global_concepts: Dict[str, str]  # concept -> description
    confidence_notes: List[str]


# ============================================================================
# STATIC SEMANTIC ENRICHER (RULE-BASED)
# ============================================================================

class StaticSemanticEnricher:
    """Deterministic semantic enrichment based on naming patterns."""
    
    # Pattern definitions
    TEMPORAL_PATTERNS = [
        r'.*_(at|date|time|timestamp|datetime)$',
        r'^(created|updated|modified|deleted|published|expired)_',
        r'.*(year|month|day|hour|minute|second).*'
    ]
    
    FINANCIAL_PATTERNS = [
        r'.*(price|cost|amount|revenue|profit|loss|fee|charge|payment|salary|wage).*',
        r'.*(total|subtotal|discount|tax|balance|debt|credit).*'
    ]
    
    IDENTIFIER_PATTERNS = [
        r'.*_id$',
        r'^id$',
        r'.*(uuid|guid|code|number|sku|barcode).*'
    ]
    
    QUANTITY_PATTERNS = [
        r'.*(quantity|qty|count|number|num|total).*',
        r'.*(stock|inventory|units).*'
    ]
    
    BOOLEAN_PATTERNS = [
        r'^(is|has|can|should|must|will)_.*',
        r'.*(enabled|active|deleted|archived|verified|confirmed)$'
    ]
    
    # Domain patterns
    DOMAIN_KEYWORDS = {
        'sales': ['order', 'sale', 'invoice', 'payment', 'customer', 'product'],
        'inventory': ['stock', 'warehouse', 'item', 'product', 'supplier'],
        'auth': ['user', 'role', 'permission', 'session', 'token', 'login'],
        'hr': ['employee', 'department', 'salary', 'attendance', 'leave'],
        'finance': ['transaction', 'account', 'ledger', 'payment', 'invoice'],
        'logistics': ['shipment', 'delivery', 'carrier', 'tracking', 'warehouse'],
        'crm': ['customer', 'contact', 'lead', 'opportunity', 'campaign']
    }
    
    def __init__(self, schema_info: SchemaInfo):
        self.schema_info = schema_info
        self.table_map = {t.name: t for t in schema_info.tables}
    
    def enrich(self) -> Dict[str, Any]:
        """Perform static semantic enrichment."""
        enrichment = {
            'table_semantics': {},
            'field_categories': {},
            'join_graph': [],
            'domain_detection': {},
            'table_types': {},
            'derived_fields': {},
            'time_series_tables': []
        }
        
        # Process each table
        for table in self.schema_info.tables:
            table_name = table.name
            
            # Detect table type
            table_type = self._detect_table_type(table)
            enrichment['table_types'][table_name] = table_type
            
            # Detect domain
            domain = self._detect_domain(table)
            enrichment['domain_detection'][table_name] = domain
            
            # Categorize fields
            field_cats = {}
            for col in table.columns:
                category = self._categorize_field(col)
                field_cats[col.name] = category
            enrichment['field_categories'][table_name] = field_cats
            
            # Detect time series
            if self._is_time_series_table(table, field_cats):
                enrichment['time_series_tables'].append(table_name)
            
            # Detect simple derived fields
            derived = self._detect_derived_fields(table)
            if derived:
                enrichment['derived_fields'][table_name] = derived
        
        # Build join graph
        enrichment['join_graph'] = self._build_join_graph()
        
        return enrichment
    
    def _categorize_field(self, col: ColumnInfo) -> str:
        """Categorize a field based on name and type."""
        col_name_lower = col.name.lower()
        col_type_lower = col.data_type.lower()
        
        # Check patterns
        if any(re.match(p, col_name_lower) for p in self.TEMPORAL_PATTERNS):
            return 'temporal'
        
        if any(re.match(p, col_name_lower) for p in self.FINANCIAL_PATTERNS):
            return 'financial'
        
        if any(re.match(p, col_name_lower) for p in self.IDENTIFIER_PATTERNS):
            return 'identifier'
        
        if any(re.match(p, col_name_lower) for p in self.QUANTITY_PATTERNS):
            return 'quantity'
        
        if any(re.match(p, col_name_lower) for p in self.BOOLEAN_PATTERNS):
            return 'boolean'
        
        # Type-based fallback
        if 'int' in col_type_lower or 'decimal' in col_type_lower or 'numeric' in col_type_lower:
            return 'numeric'
        
        if 'date' in col_type_lower or 'time' in col_type_lower:
            return 'temporal'
        
        if 'bool' in col_type_lower:
            return 'boolean'
        
        return 'text'
    
    def _detect_table_type(self, table: TableInfo) -> str:
        """Detect if table is fact, dimension, or reference."""
        col_names = [c.name.lower() for c in table.columns]
        
        # Check for transaction indicators
        has_temporal = any('date' in name or 'time' in name for name in col_names)
        has_amounts = any('amount' in name or 'price' in name or 'total' in name for name in col_names)
        has_quantities = any('quantity' in name or 'qty' in name for name in col_names)
        fk_count = len(table.foreign_keys)
        
        # Fact table heuristics
        if (has_temporal and has_amounts) or (has_quantities and fk_count >= 2):
            return 'fact'
        
        # Reference table heuristics
        if len(table.columns) <= 5 and 'name' in col_names:
            return 'reference'
        
        # Dimension table heuristics
        if fk_count <= 1 and len(table.columns) >= 5:
            return 'dimension'
        
        return 'unknown'
    
    def _detect_domain(self, table: TableInfo) -> str:
        """Detect business domain of table."""
        table_name_lower = table.name.lower()
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in table_name_lower for kw in keywords):
                return domain
        
        return 'general'
    
    def _is_time_series_table(self, table: TableInfo, field_cats: Dict[str, str]) -> bool:
        """Check if table is time-series data."""
        temporal_fields = [k for k, v in field_cats.items() if v == 'temporal']
        numeric_fields = [k for k, v in field_cats.items() if v in ['financial', 'quantity', 'numeric']]
        
        return len(temporal_fields) >= 1 and len(numeric_fields) >= 1
    
    def _detect_derived_fields(self, table: TableInfo) -> List[Dict[str, Any]]:
        """Detect simple derived fields."""
        derived = []
        col_map = {c.name.lower(): c for c in table.columns}
        
        # Detect revenue = quantity * price
        if 'quantity' in col_map and 'price' in col_map:
            derived.append({
                'name': 'revenue',
                'expression': 'quantity * price',
                'description': 'Total revenue calculated from quantity and price',
                'required_columns': ['quantity', 'price'],
                'confidence': 'high'
            })
        
        # Detect margin = selling_price - cost_price
        if 'selling_price' in col_map and 'cost_price' in col_map:
            derived.append({
                'name': 'margin',
                'expression': 'selling_price - cost_price',
                'description': 'Profit margin per unit',
                'required_columns': ['selling_price', 'cost_price'],
                'confidence': 'high'
            })
        
        # Detect age/duration calculations
        if 'created_at' in col_map and 'updated_at' in col_map:
            derived.append({
                'name': 'processing_time',
                'expression': 'updated_at - created_at',
                'description': 'Time elapsed between creation and update',
                'required_columns': ['created_at', 'updated_at'],
                'confidence': 'high'
            })
        
        return derived
    
    def _build_join_graph(self) -> List[Dict[str, Any]]:
        """Build join path graph from foreign keys."""
        join_paths = []
        
        for table in self.schema_info.tables:
            for col_name, ref in table.foreign_keys.items():
                if '.' in ref:
                    ref_table, ref_col = ref.split('.')
                    
                    # Determine join type
                    col_info = next((c for c in table.columns if c.name == col_name), None)
                    if col_info and col_info.primary_key:
                        join_type = 'one-to-one'
                    else:
                        join_type = 'many-to-one'
                    
                    join_paths.append({
                        'from_table': table.name,
                        'to_table': ref_table,
                        'join_type': join_type,
                        'via_columns': [(col_name, ref_col)],
                        'description': f'{table.name} references {ref_table}'
                    })
        
        # Detect many-to-many through junction tables
        for table in self.schema_info.tables:
            if len(table.foreign_keys) == 2:
                refs = list(table.foreign_keys.values())
                if len(refs) == 2:
                    ref1 = refs[0].split('.')[0] if '.' in refs[0] else refs[0]
                    ref2 = refs[1].split('.')[0] if '.' in refs[1] else refs[1]
                    
                    join_paths.append({
                        'from_table': ref1,
                        'to_table': ref2,
                        'join_type': 'many-to-many',
                        'via_columns': [],
                        'intermediate_tables': [table.name],
                        'description': f'{ref1} and {ref2} are related through {table.name}'
                    })
        
        return join_paths


# ============================================================================
# LLM SEMANTIC ENRICHER (CREATIVE LAYER)
# ============================================================================

class LLMSemanticEnricher:
    """LLM-powered creative semantic enrichment."""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.2,
            max_tokens=8000
        )
        
        self.system_prompt = """You are a database schema analyst. 
Generate semantic enrichment for database schemas.

CRITICAL RULES:
1. Only infer from actual table/column names and relationships
2. Do NOT hallucinate columns or tables that don't exist
3. Mark uncertain inferences with confidence: low
4. Focus on business concepts that can be derived from names
5. Suggest only valid SQL expressions for derived fields
6. Return ONLY valid JSON, no markdown, no explanations"""
    
    def enrich(
        self, 
        structural_metadata: Dict[str, Any],
        static_enrichment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform LLM-based semantic enrichment."""
        
        # Prepare compact metadata for LLM
        compact_meta = self._prepare_compact_metadata(
            structural_metadata, 
            static_enrichment
        )
        
        prompt = f"""Based on this schema metadata, generate semantic enrichment.

Schema Metadata:
{json.dumps(compact_meta, indent=2)}

Generate JSON with:
{{
  "table_enrichments": {{
    "table_name": {{
      "business_purpose": "clear purpose",
      "use_cases": ["use case 1", "use case 2"],
      "field_meanings": {{
        "column_name": {{
          "meaning": "business meaning",
          "synonyms": ["synonym1", "synonym2"],
          "units": "units if applicable"
        }}
      }},
      "suggested_metrics": [
        {{
          "name": "metric_name",
          "expression": "SQL expression",
          "description": "what it measures",
          "confidence": "high/medium/low"
        }}
      ]
    }}
  }},
  "domain_concepts": {{
    "concept_name": "description"
  }},
  "join_interpretations": [
    {{
      "from": "table1",
      "to": "table2",
      "business_meaning": "why these tables are related"
    }}
  ],
  "confidence_notes": ["note about uncertain inferences"]
}}

Remember: Only use columns/tables that exist in the schema."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            
            # Clean response
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = re.sub(r'```json\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
            
            # Parse JSON
            enrichment = json.loads(content)
            
            return enrichment
            
        except Exception as e:
            print(f"Warning: LLM enrichment failed: {e}")
            return {}
    
    def _prepare_compact_metadata(
        self,
        structural: Dict[str, Any],
        static: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare compact metadata for LLM."""
        compact = {
            'tables': []
        }
        
        # Get table info
        for table_name, table_data in structural.items():
            if not isinstance(table_data, dict):
                continue
            
            table_info = {
                'name': table_name,
                'type': static.get('table_types', {}).get(table_name, 'unknown'),
                'domain': static.get('domain_detection', {}).get(table_name, 'general'),
                'columns': []
            }
            
            # Add column info
            for col in table_data.get('columns', []):
                col_name = col.get('name', '')
                col_info = {
                    'name': col_name,
                    'type': col.get('type', ''),
                    'category': static.get('field_categories', {}).get(table_name, {}).get(col_name, 'unknown'),
                    'nullable': col.get('nullable', True),
                    'primary_key': col.get('primary_key', False)
                }
                table_info['columns'].append(col_info)
            
            # Add foreign keys
            fks = table_data.get('foreign_keys', {})
            if fks:
                table_info['foreign_keys'] = fks
            
            compact['tables'].append(table_info)
        
        # Add join graph
        compact['join_paths'] = static.get('join_graph', [])
        
        return compact


# ============================================================================
# VALIDATION LAYER (GUARDRAIL)
# ============================================================================

class SemanticValidator:
    """Validate LLM-generated semantic enrichment."""
    
    def __init__(self, schema_info: SchemaInfo):
        self.schema_info = schema_info
        self.table_names = {t.name.lower() for t in schema_info.tables}
        self.column_map = {}
        
        for table in schema_info.tables:
            self.column_map[table.name.lower()] = {
                c.name.lower() for c in table.columns
            }
    
    def validate(
        self,
        llm_enrichment: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate and clean LLM enrichment."""
        
        validated = {
            'table_enrichments': {},
            'domain_concepts': {},
            'join_interpretations': [],
            'confidence_notes': []
        }
        
        validation_errors = []
        
        # Validate table enrichments
        for table_name, enrichment in llm_enrichment.get('table_enrichments', {}).items():
            if table_name.lower() not in self.table_names:
                validation_errors.append(f"Table {table_name} does not exist")
                continue
            
            validated_table = self._validate_table_enrichment(
                table_name,
                enrichment,
                validation_errors
            )
            
            if validated_table:
                validated['table_enrichments'][table_name] = validated_table
        
        # Validate domain concepts (low risk)
        validated['domain_concepts'] = llm_enrichment.get('domain_concepts', {})
        
        # Validate join interpretations
        for join in llm_enrichment.get('join_interpretations', []):
            from_table = join.get('from', '').lower()
            to_table = join.get('to', '').lower()
            
            if from_table in self.table_names and to_table in self.table_names:
                validated['join_interpretations'].append(join)
            else:
                validation_errors.append(
                    f"Join interpretation references non-existent tables: {from_table} -> {to_table}"
                )
        
        # Merge confidence notes
        validated['confidence_notes'] = (
            llm_enrichment.get('confidence_notes', []) + 
            validation_errors
        )
        
        return validated, validation_errors
    
    def _validate_table_enrichment(
        self,
        table_name: str,
        enrichment: Dict[str, Any],
        errors: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Validate single table enrichment."""
        
        validated = {
            'business_purpose': enrichment.get('business_purpose', ''),
            'use_cases': enrichment.get('use_cases', []),
            'field_meanings': {},
            'suggested_metrics': []
        }
        
        table_columns = self.column_map.get(table_name.lower(), set())
        
        # Validate field meanings
        for col_name, meaning in enrichment.get('field_meanings', {}).items():
            if col_name.lower() in table_columns:
                validated['field_meanings'][col_name] = meaning
            else:
                errors.append(f"Column {col_name} does not exist in table {table_name}")
        
        # Validate metrics
        for metric in enrichment.get('suggested_metrics', []):
            is_valid, error = self._validate_metric(
                metric,
                table_columns,
                table_name
            )
            
            if is_valid:
                validated['suggested_metrics'].append(metric)
            else:
                errors.append(error)
        
        return validated
    
    def _validate_metric(
        self,
        metric: Dict[str, Any],
        table_columns: Set[str],
        table_name: str
    ) -> Tuple[bool, str]:
        """Validate a suggested metric."""
        
        expression = metric.get('expression', '')
        
        # Extract column names from expression
        # Simple pattern matching for column names
        potential_cols = re.findall(r'\b([a-z_][a-z0-9_]*)\b', expression.lower())
        
        # Check if all referenced columns exist
        for col in potential_cols:
            if col not in table_columns and col not in {'sum', 'avg', 'count', 'max', 'min', 'distinct', 'case', 'when', 'then', 'else', 'end', 'as', 'and', 'or', 'not', 'null', 'true', 'false'}:
                return False, f"Metric references non-existent column: {col} in table {table_name}"
        
        return True, ""


# ============================================================================
# SEMANTIC PROFILE BUILDER
# ============================================================================

class SemanticProfileBuilder:
    """Build final semantic profile from all enrichment layers."""
    
    def __init__(
        self,
        schema_info: SchemaInfo,
        static_enrichment: Dict[str, Any],
        llm_enrichment: Dict[str, Any]
    ):
        self.schema_info = schema_info
        self.static_enrichment = static_enrichment
        self.llm_enrichment = llm_enrichment
    
    def build(self) -> SemanticSchema:
        """Build complete semantic schema."""
        
        # Build table semantics
        table_semantics = []
        for table in self.schema_info.tables:
            table_sem = self._build_table_semantics(table)
            table_semantics.append(table_sem)
        
        # Build join paths
        join_paths = self._build_join_paths()
        
        # Build domain groupings
        domain_groupings = self._build_domain_groupings()
        
        # Build global concepts
        global_concepts = self.llm_enrichment.get('domain_concepts', {})
        
        # Confidence notes
        confidence_notes = self.llm_enrichment.get('confidence_notes', [])
        
        # Create semantic schema
        semantic_schema = SemanticSchema(
            schema_name=f"schema_{len(self.schema_info.tables)}_tables",
            version="1.0",
            created_at=datetime.now().isoformat(),
            structural_metadata=self._get_structural_metadata(),
            table_semantics=table_semantics,
            join_paths=join_paths,
            domain_groupings=domain_groupings,
            global_concepts=global_concepts,
            confidence_notes=confidence_notes
        )
        
        return semantic_schema
    
    def _build_table_semantics(self, table: TableInfo) -> TableSemantics:
        """Build semantic metadata for a table."""
        
        table_name = table.name
        
        # Get static data
        table_type = self.static_enrichment.get('table_types', {}).get(table_name, 'unknown')
        domain = self.static_enrichment.get('domain_detection', {}).get(table_name, 'general')
        
        # Get LLM data
        llm_table = self.llm_enrichment.get('table_enrichments', {}).get(table_name, {})
        business_purpose = llm_table.get('business_purpose', f'Stores {table_name} data')
        use_cases = llm_table.get('use_cases', [])
        
        # Build field semantics
        field_semantics = []
        field_cats = self.static_enrichment.get('field_categories', {}).get(table_name, {})
        llm_meanings = llm_table.get('field_meanings', {})
        
        for col in table.columns:
            field_sem = FieldSemantics(
                column_name=col.name,
                business_meaning=llm_meanings.get(col.name, {}).get('meaning', col.name),
                synonyms=llm_meanings.get(col.name, {}).get('synonyms', []),
                data_category=field_cats.get(col.name, 'unknown'),
                is_metric=field_cats.get(col.name) in ['financial', 'quantity', 'numeric'],
                metric_type='sum' if field_cats.get(col.name) in ['financial', 'quantity'] else None,
                units=llm_meanings.get(col.name, {}).get('units'),
                tags=[]
            )
            field_semantics.append(field_sem)
        
        # Build derived fields
        derived_fields = []
        static_derived = self.static_enrichment.get('derived_fields', {}).get(table_name, [])
        llm_metrics = llm_table.get('suggested_metrics', [])
        
        for d in static_derived:
            derived_fields.append(DerivedField(
                name=d['name'],
                expression=d['expression'],
                description=d['description'],
                required_columns=d['required_columns'],
                output_type='numeric',
                confidence=d.get('confidence', 'high')
            ))
        
        for m in llm_metrics:
            derived_fields.append(DerivedField(
                name=m.get('name', 'unnamed_metric'),
                expression=m.get('expression', ''),
                description=m.get('description', ''),
                required_columns=[],
                output_type='numeric',
                confidence=m.get('confidence', 'medium')
            ))
        
        # Common metrics
        common_metrics = [d.name for d in derived_fields]
        
        # Related tables
        related_tables = list(set(
            ref.split('.')[0] for ref in table.foreign_keys.values()
        ))
        
        return TableSemantics(
            table_name=table_name,
            table_type=table_type,
            business_domain=domain,
            business_purpose=business_purpose,
            use_cases=use_cases,
            field_semantics=field_semantics,
            derived_fields=derived_fields,
            common_metrics=common_metrics,
            related_tables=related_tables,
            tags=[table_type, domain]
        )
    
    def _build_join_paths(self) -> List[JoinPath]:
        """Build join paths from static enrichment."""
        join_paths = []
        
        for join in self.static_enrichment.get('join_graph', []):
            join_path = JoinPath(
                from_table=join['from_table'],
                to_table=join['to_table'],
                join_type=join['join_type'],
                via_columns=join['via_columns'],
                intermediate_tables=join.get('intermediate_tables', []),
                description=join.get('description', '')
            )
            join_paths.append(join_path)
        
        return join_paths
    
    def _build_domain_groupings(self) -> Dict[str, List[str]]:
        """Build domain groupings."""
        groupings = {}
        
        for table_name, domain in self.static_enrichment.get('domain_detection', {}).items():
            if domain not in groupings:
                groupings[domain] = []
            groupings[domain].append(table_name)
        
        return groupings
    
    def _get_structural_metadata(self) -> Dict[str, Any]:
        """Get structural metadata."""
        return {
            'table_count': len(self.schema_info.tables),
            'total_columns': sum(len(t.columns) for t in self.schema_info.tables),
            'total_foreign_keys': sum(len(t.foreign_keys) for t in self.schema_info.tables)
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class SemanticEnrichmentPipeline:
    """Complete semantic enrichment pipeline."""
    
    def __init__(
        self,
        schema_info: SchemaInfo,
        api_key: str,
        workspace_dir: Path
    ):
        self.schema_info = schema_info
        self.api_key = api_key
        self.workspace_dir = workspace_dir
    
    def run(self) -> SemanticSchema:
        """Run complete enrichment pipeline."""
        
        print("🔍 Stage 1: Static semantic enrichment...")
        static_enricher = StaticSemanticEnricher(self.schema_info)
        static_enrichment = static_enricher.enrich()
        
        print("🤖 Stage 2: LLM semantic enrichment...")
        llm_enricher = LLMSemanticEnricher(self.api_key)
        
        # Prepare structural metadata
        structural_metadata = {}
        for table in self.schema_info.tables:
            # Convert columns to dict manually
            columns_data = []
            for col in table.columns:
                col_dict = {
                    'name': col.name,
                    'data_type': col.data_type,
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'foreign_key': col.foreign_key,
                    'default': col.default
                }
                columns_data.append(col_dict)
            
            structural_metadata[table.name] = {
                'columns': columns_data,
                'primary_keys': table.primary_keys,
                'foreign_keys': table.foreign_keys,
                'indexes': table.indexes
            }
        
        llm_enrichment = llm_enricher.enrich(structural_metadata, static_enrichment)
        
        print("✅ Stage 3: Validating LLM enrichment...")
        validator = SemanticValidator(self.schema_info)
        validated_enrichment, errors = validator.validate(llm_enrichment)
        
        if errors:
            print(f"⚠️  Found {len(errors)} validation issues")
            for error in errors[:5]:
                print(f"   - {error}")
        
        print("📦 Stage 4: Building semantic profile...")
        builder = SemanticProfileBuilder(
            self.schema_info,
            static_enrichment,
            validated_enrichment
        )
        semantic_schema = builder.build()
        
        print("💾 Stage 5: Saving semantic profile...")
        self._save_semantic_profile(semantic_schema)
        
        print("✨ Semantic enrichment complete!")
        return semantic_schema
    
    def _save_semantic_profile(self, semantic_schema: SemanticSchema) -> None:
        """Save semantic profile to workspace."""
        profile_path = self.workspace_dir / "semantic_profile.json"
        
        # Convert to dict manually to avoid dataclass issues
        profile_dict = {
            'schema_name': semantic_schema.schema_name,
            'version': semantic_schema.version,
            'created_at': semantic_schema.created_at,
            'structural_metadata': semantic_schema.structural_metadata,
            'table_semantics': self._convert_table_semantics(semantic_schema.table_semantics),
            'join_paths': self._convert_join_paths(semantic_schema.join_paths),
            'domain_groupings': semantic_schema.domain_groupings,
            'global_concepts': semantic_schema.global_concepts,
            'confidence_notes': semantic_schema.confidence_notes
        }
        
        # Save
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)
        
        print(f"   Saved to: {profile_path}")
    
    def _convert_table_semantics(self, table_semantics: List[TableSemantics]) -> List[Dict[str, Any]]:
        """Convert TableSemantics to dict."""
        result = []
        for ts in table_semantics:
            ts_dict = {
                'table_name': ts.table_name,
                'table_type': ts.table_type,
                'business_domain': ts.business_domain,
                'business_purpose': ts.business_purpose,
                'use_cases': ts.use_cases,
                'field_semantics': self._convert_field_semantics(ts.field_semantics),
                'derived_fields': self._convert_derived_fields(ts.derived_fields),
                'common_metrics': ts.common_metrics,
                'related_tables': ts.related_tables,
                'tags': ts.tags
            }
            result.append(ts_dict)
        return result
    
    def _convert_field_semantics(self, field_semantics: List[FieldSemantics]) -> List[Dict[str, Any]]:
        """Convert FieldSemantics to dict."""
        result = []
        for fs in field_semantics:
            fs_dict = {
                'column_name': fs.column_name,
                'business_meaning': fs.business_meaning,
                'synonyms': fs.synonyms,
                'data_category': fs.data_category,
                'is_metric': fs.is_metric,
                'metric_type': fs.metric_type,
                'units': fs.units,
                'tags': fs.tags
            }
            result.append(fs_dict)
        return result
    
    def _convert_derived_fields(self, derived_fields: List[DerivedField]) -> List[Dict[str, Any]]:
        """Convert DerivedField to dict."""
        result = []
        for df in derived_fields:
            df_dict = {
                'name': df.name,
                'expression': df.expression,
                'description': df.description,
                'required_columns': df.required_columns,
                'output_type': df.output_type,
                'confidence': df.confidence
            }
            result.append(df_dict)
        return result
    
    def _convert_join_paths(self, join_paths: List[JoinPath]) -> List[Dict[str, Any]]:
        """Convert JoinPath to dict."""
        result = []
        for jp in join_paths:
            jp_dict = {
                'from_table': jp.from_table,
                'to_table': jp.to_table,
                'join_type': jp.join_type,
                'via_columns': jp.via_columns,
                'intermediate_tables': jp.intermediate_tables,
                'description': jp.description
            }
            result.append(jp_dict)
        return result


# ============================================================================
# SEMANTIC DOCUMENT GENERATOR (FOR VECTOR DB)
# ============================================================================

class SemanticDocumentGenerator:
    """Generate rich semantic documents for embedding."""
    
    def __init__(self, semantic_schema: SemanticSchema):
        self.semantic_schema = semantic_schema
    
    def generate_documents(self) -> List[Dict[str, Any]]:
        """Generate semantic documents for vector embedding."""
        documents = []
        
        # Generate per-table documents
        for table_sem in self.semantic_schema.table_semantics:
            doc = self._generate_table_document(table_sem)
            documents.append(doc)
        
        # Generate domain documents
        for domain, tables in self.semantic_schema.domain_groupings.items():
            doc = self._generate_domain_document(domain, tables)
            documents.append(doc)
        
        # Generate join path documents
        for join in self.semantic_schema.join_paths:
            doc = self._generate_join_document(join)
            documents.append(doc)
        
        return documents
    
    def _generate_table_document(self, table_sem: TableSemantics) -> Dict[str, Any]:
        """Generate rich document for a table."""
        
        # Build comprehensive text
        lines = []
        
        # Header with metadata
        lines.append(f"# Table: {table_sem.table_name}")
        lines.append(f"Type: {table_sem.table_type}")
        lines.append(f"Domain: {table_sem.business_domain}")
        lines.append(f"Purpose: {table_sem.business_purpose}")
        lines.append("")
        
        # Use cases
        if table_sem.use_cases:
            lines.append("## Use Cases")
            for uc in table_sem.use_cases:
                lines.append(f"- {uc}")
            lines.append("")
        
        # Fields with semantics
        lines.append("## Fields")
        for field in table_sem.field_semantics:
            field_line = f"- {field.column_name}: {field.business_meaning}"
            
            if field.synonyms:
                field_line += f" (also: {', '.join(field.synonyms)})"
            
            if field.units:
                field_line += f" [{field.units}]"
            
            field_line += f" [{field.data_category}]"
            
            lines.append(field_line)
        lines.append("")
        
        # Derived fields
        if table_sem.derived_fields:
            lines.append("## Derived Metrics")
            for df in table_sem.derived_fields:
                lines.append(f"- {df.name}: {df.description}")
                lines.append(f"  Expression: {df.expression}")
            lines.append("")
        
        # Common metrics
        if table_sem.common_metrics:
            lines.append("## Common Metrics")
            lines.append(f"Available: {', '.join(table_sem.common_metrics)}")
            lines.append("")
        
        # Related tables
        if table_sem.related_tables:
            lines.append("## Related Tables")
            lines.append(f"Connected to: {', '.join(table_sem.related_tables)}")
            lines.append("")
        
        # Searchable keywords
        keywords = self._extract_keywords(table_sem)
        lines.append("## Search Keywords")
        lines.append(f"{', '.join(keywords)}")
        
        content = "\n".join(lines)
        
        return {
            "table": table_sem.table_name,
            "content": content,
            "type": "table_semantic",
            "metadata": {
                "table_type": table_sem.table_type,
                "domain": table_sem.business_domain,
                "tags": table_sem.tags
            }
        }
    
    def _generate_domain_document(self, domain: str, tables: List[str]) -> Dict[str, Any]:
        """Generate document for domain grouping."""
        
        lines = []
        lines.append(f"# Domain: {domain}")
        lines.append("")
        lines.append(f"Tables in this domain: {', '.join(tables)}")
        lines.append("")
        
        # Get domain concept if available
        if domain in self.semantic_schema.global_concepts:
            lines.append(f"Description: {self.semantic_schema.global_concepts[domain]}")
        
        content = "\n".join(lines)
        
        return {
            "table": f"_domain_{domain}",
            "content": content,
            "type": "domain",
            "metadata": {
                "domain": domain,
                "table_count": len(tables)
            }
        }
    
    def _generate_join_document(self, join: JoinPath) -> Dict[str, Any]:
        """Generate document for join path."""
        
        lines = []
        lines.append(f"# Join Path: {join.from_table} → {join.to_table}")
        lines.append(f"Type: {join.join_type}")
        lines.append(f"Description: {join.description}")
        lines.append("")
        
        if join.via_columns:
            lines.append("Join Columns:")
            for left, right in join.via_columns:
                lines.append(f"- {join.from_table}.{left} = {join.to_table}.{right}")
        
        if join.intermediate_tables:
            lines.append("")
            lines.append(f"Via: {' → '.join(join.intermediate_tables)}")
        
        content = "\n".join(lines)
        
        return {
            "table": f"_join_{join.from_table}_{join.to_table}",
            "content": content,
            "type": "join_path",
            "metadata": {
                "from_table": join.from_table,
                "to_table": join.to_table,
                "join_type": join.join_type
            }
        }
    
    def _extract_keywords(self, table_sem: TableSemantics) -> List[str]:
        """Extract searchable keywords from table semantics."""
        keywords = set()
        
        # Table name variations
        keywords.add(table_sem.table_name)
        keywords.add(table_sem.table_name.replace('_', ' '))
        
        # Domain and type
        keywords.add(table_sem.business_domain)
        keywords.add(table_sem.table_type)
        
        # Field names and synonyms
        for field in table_sem.field_semantics:
            keywords.add(field.column_name)
            keywords.update(field.synonyms)
        
        # Metric names
        keywords.update(table_sem.common_metrics)
        
        # Related tables
        keywords.update(table_sem.related_tables)
        
        # Tags
        keywords.update(table_sem.tags)
        
        return list(keywords)