"""
sqlgenx/core/deterministic_validator.py

Deterministic semantic validation without LLM.
Validates NL queries using semantic search and fact checking.
"""

from typing import Tuple, List, Dict, Any, Set
from pathlib import Path
import json
import re
import sys
from sqlgenx.core.vector_store import VectorStore


class DebugLogger:
    """Simple debug logger that writes directly to stderr."""

    @staticmethod
    def log(message: str):
        """Write debug message to stderr (bypasses progress bar)."""
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()


class DeterministicSemanticValidator:
    """
    Validates NL queries using semantic search, NO LLM.

    Validation logic:
    1. Search semantic profile for query keywords
    2. Check if retrieved tables can answer the query
    3. Extract required entities (temporal, aggregation, etc.)
    4. Verify entity availability via semantic metadata
    5. Return: (is_valid, reason, relevant_contexts)
    """

    # Query intent patterns
    TEMPORAL_PATTERNS = [
        r'\b(daily|weekly|monthly|yearly|quarterly)\b',
        r'\b(today|yesterday|last\s+(week|month|year|quarter))\b',
        r'\b(trend|over\s+time|time\s+series)\b',
        r'\b(consecutive|sequential|in\s+a\s+row)\b',
        r'\b(period|date|time|when)\b',
        r'\b(decline|increase|growth|decrease|rising|falling)\b'
    ]

    AGGREGATION_PATTERNS = [
        r'\b(total|sum|average|avg|count|max|min|maximum|minimum)\b',
        r'\b(revenue|sales|profit|income|cost|expense)\b',
        r'\b(group\s+by|per|by\s+category|by\s+product)\b',
        r'\b(top|bottom|highest|lowest|best|worst)\b'
    ]

    COMPARISON_PATTERNS = [
        r'\b(compare|versus|vs|difference|between)\b',
        r'\b(more\s+than|less\s+than|greater|smaller)\b',
        r'\b(rank|ranking|order\s+by)\b'
    ]

    JOIN_PATTERNS = [
        r'\b(customer.*product|product.*customer)\b',
        r'\b(user.*order|order.*user)\b',
        r'\b(with|that\s+have|who\s+bought)\b'
    ]

    def __init__(self, workspace_dir: Path, debug: bool = False):
        self.workspace_dir = workspace_dir
        self.vector_store = VectorStore(workspace_dir)
        self.semantic_profile = self._load_semantic_profile()
        self.debug = debug

    def _load_semantic_profile(self) -> Dict[str, Any]:
        """Load semantic profile JSON."""
        profile_path = self.workspace_dir / "semantic_profile.json"

        if not profile_path.exists():
            return {}

        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def validate(
        self,
        user_query: str,
        n_results: int = 5,
        min_relevance: float = 0.3
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Validate NL query deterministically using semantic search.

        Args:
            user_query: User's natural language query
            n_results: Number of semantic documents to retrieve
            min_relevance: Minimum relevance score (0-1)

        Returns:
            Tuple of (is_valid, reason, relevant_contexts)
        """

        if self.debug:
            print(f"\n[DEBUG] Validating query: {user_query}")
            print(f"[DEBUG] Search parameters: n_results={n_results}, min_relevance={min_relevance}")

        # Step 1: Extract query intent
        intent = self._extract_query_intent(user_query)

        if self.debug:
            print(f"\n[DEBUG] Query Intent:")
            print(f"  - Needs temporal: {intent['needs_temporal']}")
            print(f"  - Needs aggregation: {intent['needs_aggregation']}")
            print(f"  - Needs comparison: {intent['needs_comparison']}")
            print(f"  - Needs joins: {intent['needs_joins']}")
            print(f"  - Keywords: {list(intent['keywords'])[:10]}")

        # Step 2: Semantic search
        retrieved_contexts = self.vector_store.retrieve_context(
            query=user_query,
            n_results=n_results
        )

        if self.debug:
            print(f"\n[DEBUG] Semantic Search Results:")
            print(f"  Total retrieved: {len(retrieved_contexts)}")
            for i, ctx in enumerate(retrieved_contexts):
                table = ctx.get('table', 'unknown')
                score = ctx.get('relevance_score', 0)
                ctx_type = ctx.get('type', 'unknown')
                print(f"  {i+1}. {table} (score: {score:.3f}, type: {ctx_type})")

        if not retrieved_contexts:
            return False, "No relevant tables found in schema for this query", []

        # Step 3: Filter by relevance threshold
        relevant_contexts = [
            ctx for ctx in retrieved_contexts
            if ctx.get('relevance_score', 0) >= min_relevance
        ]

        if self.debug:
            print(f"\n[DEBUG] After Relevance Filtering (min: {min_relevance}):")
            print(f"  Kept: {len(relevant_contexts)} / {len(retrieved_contexts)}")
            if not relevant_contexts:
                print("  ⚠️  All results below threshold!")
                print(f"  Highest score: {max([ctx.get('relevance_score', 0) for ctx in retrieved_contexts]):.3f}")

        if not relevant_contexts:
            highest_score = max([ctx.get('relevance_score', 0) for ctx in retrieved_contexts]) if retrieved_contexts else 0
            return False, (
                f"No tables with sufficient relevance (min: {min_relevance}, highest: {highest_score:.3f}). "
                f"Try enriching schema or lowering threshold."
            ), []

        # Step 4: Build capability map from retrieved contexts
        capabilities = self._build_capability_map(relevant_contexts)

        if self.debug:
            print(f"\n[DEBUG] Detected Capabilities:")
            print(f"  - Has temporal: {capabilities['has_temporal']}")
            print(f"  - Temporal via join: {capabilities['temporal_via_join']}")
            print(f"  - Temporal tables: {capabilities['temporal_tables']}")
            print(f"  - Has aggregation: {capabilities['has_aggregation']}")
            print(f"  - Aggregatable fields: {len(capabilities['aggregatable_fields'])}")
            print(f"  - Derived metrics: {len(capabilities['derived_metrics'])}")
            print(f"  - Available tables: {capabilities['available_tables']}")
            print(f"  - Join paths: {len(capabilities['join_paths'])}")

        # Step 5: Check if capabilities satisfy intent
        is_valid, reason = self._check_capabilities(intent, capabilities)

        if self.debug:
            print(f"\n[DEBUG] Capability Check:")
            print(f"  Valid: {is_valid}")
            print(f"  Reason: {reason}")

        if not is_valid:
            return False, reason, []

        # Step 6: Return only relevant tables
        filtered_contexts = self._filter_relevant_contexts(
            relevant_contexts,
            intent,
            capabilities
        )

        if self.debug:
            print(f"\n[DEBUG] Final Filtered Contexts:")
            print(f"  Returning: {len(filtered_contexts)} tables")
            for ctx in filtered_contexts:
                print(f"  - {ctx.get('table', 'unknown')}")

        return True, "Query can be answered with available schema", filtered_contexts

    def _extract_query_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Extract what the query needs (temporal, aggregation, joins, etc.)
        """
        query_lower = user_query.lower()

        intent = {
            'needs_temporal': False,
            'needs_aggregation': False,
            'needs_comparison': False,
            'needs_joins': False,
            'keywords': set(),
            'entities': set()
        }

        # Check temporal requirements
        for pattern in self.TEMPORAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                intent['needs_temporal'] = True
                break

        # Check aggregation requirements
        for pattern in self.AGGREGATION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                intent['needs_aggregation'] = True
                break

        # Check comparison requirements
        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                intent['needs_comparison'] = True
                break

        # Check join requirements
        for pattern in self.JOIN_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                intent['needs_joins'] = True
                break

        # Extract keywords (simple tokenization)
        words = re.findall(r'\b[a-z_]{3,}\b', query_lower)
        intent['keywords'] = set(words)

        # Extract potential entities (capitalized words, numbers)
        entities = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', user_query)
        intent['entities'] = set(entities)

        return intent

    def _build_capability_map(
        self,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build capability map from retrieved semantic contexts.

        Returns what operations are possible across retrieved tables.
        """
        capabilities = {
            'has_temporal': False,
            'temporal_via_join': False,
            'temporal_tables': set(),
            'has_aggregation': False,
            'aggregatable_fields': [],
            'join_paths': [],
            'available_tables': set(),
            'derived_metrics': []
        }

        for ctx in contexts:
            content = ctx.get('content', '')
            table = ctx.get('table', '')

            if not table or table.startswith('_'):
                continue

            capabilities['available_tables'].add(table)

            # Check for temporal capabilities in the content
            has_temporal_fields = False
            has_aggregatable_fields = False
            aggregatable_fields = []
            
            # Check for temporal indicators in various forms
            if 'temporal' in content.lower() or 'date' in content.lower() or 'time' in content.lower() or 'created_at' in content.lower() or 'updated_at' in content.lower():
                has_temporal_fields = True
                capabilities['temporal_tables'].add(table)
                
            # Look for numeric/financial fields that can be aggregated
            if 'financial' in content.lower() or 'quantity' in content.lower() or 'numeric' in content.lower():
                has_aggregatable_fields = True
            
            # Extract field information more intelligently
            # Check for fields with categories like financial, quantity, numeric
            lines = content.split('\n')
            for line in lines:
                if 'financial' in line.lower() or 'quantity' in line.lower() or 'numeric' in line.lower():
                    # Try to extract field names from lines like "- field_name: description [financial]"
                    field_match = re.search(r'-\s*(\w+):\s*.*?\[(\w+)\]', line)
                    if field_match:
                        field_name, field_type = field_match.groups()
                        if field_type.lower() in ['financial', 'quantity', 'numeric']:
                            aggregatable_fields.append({
                                'table': table,
                                'field': field_name
                            })
                            has_aggregatable_fields = True
                            
            # Check for derived metrics in the content
            if 'Available Metrics:' in content or 'Derived Metrics' in content:
                has_aggregatable_fields = True
                # Extract metric names
                in_metrics_section = False
                for line in lines:
                    if 'Available Metrics:' in line or 'Derived Metrics' in line:
                        in_metrics_section = True
                        continue
                    if in_metrics_section and line.strip().startswith('- '):
                        metric_match = re.search(r'-\s*(\w+):\s*', line)
                        if metric_match:
                            capabilities['derived_metrics'].append({
                                'table': table,
                                'metric': metric_match.group(1)
                            })
            
            # Look for related/connected tables in the content
            if 'Related to:' in content:
                related_match = re.search(r'Related to:\s*(.+)', content)
                if related_match:
                    related_tables = [t.strip() for t in related_match.group(1).split(',') if t.strip()]
                    for rel_table in related_tables:
                        if rel_table and rel_table in capabilities['available_tables']:
                            capabilities['join_paths'].append({
                                'from': table,
                                'to': rel_table
                            })
            
            # Also check for "Connected to" patterns
            connected_match = re.search(r'Connected to:\s*(.+)', content)
            if connected_match:
                related_tables = [t.strip() for t in connected_match.group(1).split(',') if t.strip()]
                for rel_table in related_tables:
                    if rel_table and rel_table in capabilities['available_tables']:
                        capabilities['join_paths'].append({
                            'from': table,
                            'to': rel_table
                        })

            # Update main flags based on findings
            if has_temporal_fields:
                capabilities['has_temporal'] = True
                
            if has_aggregatable_fields:
                capabilities['has_aggregation'] = True
                capabilities['aggregatable_fields'].extend(aggregatable_fields)

        return capabilities

    def _extract_section(self, content: str, section_header: str) -> str:
        """Extract a section from semantic document."""
        parts = content.split(section_header)
        if len(parts) < 2:
            return ""

        # Get section until next ## or end
        section = parts[1].split('\n##')[0]
        return section

    def _check_capabilities(
        self,
        intent: Dict[str, Any],
        capabilities: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if capabilities satisfy query intent.

        Returns: (is_valid, reason)
        """

        # Check temporal requirement
        if intent['needs_temporal']:
            if not capabilities['has_temporal'] and not capabilities['temporal_via_join']:
                return False, (
                    "Query requires temporal data (dates/times) but retrieved tables "
                    "do not have direct or indirect access to temporal columns"
                )

        # Check aggregation requirement
        if intent['needs_aggregation']:
            if not capabilities['has_aggregation']:
                return False, (
                    "Query requires aggregations (sum, average, count, etc.) but "
                    "retrieved tables do not have aggregatable numeric fields"
                )

        # Check if we have any tables at all
        if not capabilities['available_tables']:
            return False, "No relevant tables found in schema"

        # Check for join requirements
        if intent['needs_joins']:
            if len(capabilities['available_tables']) < 2 and not capabilities['join_paths']:
                return False, (
                    "Query requires joining multiple tables but no join paths found"
                )

        # If all checks pass
        return True, "Query requirements satisfied by schema capabilities"

    def _filter_relevant_contexts(
        self,
        contexts: List[Dict[str, Any]],
        intent: Dict[str, Any],
        capabilities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter contexts to only include tables needed for this query.

        Strategy:
        1. If temporal needed, prioritize tables with temporal access
        2. If aggregation needed, prioritize tables with aggregatable fields
        3. Include join path tables
        4. Limit to top 3-5 most relevant tables
        """

        scored_contexts = []

        for ctx in contexts:
            table = ctx.get('table', '')
            content = ctx.get('content', '')
            relevance = ctx.get('relevance_score', 0.0)

            # Skip non-table documents
            if table.startswith('_'):
                continue

            # Base score from relevance
            score = relevance

            # Boost temporal tables if temporal needed
            if intent['needs_temporal']:
                if table in capabilities['temporal_tables']:
                    score += 0.3
                elif 'temporal' in content.lower():
                    score += 0.1

            # Boost aggregatable tables if aggregation needed
            if intent['needs_aggregation']:
                if any(f['table'] == table for f in capabilities['aggregatable_fields']):
                    score += 0.2

            # Boost tables in join paths
            if intent['needs_joins']:
                if any(j['from'] == table or j['to'] == table for j in capabilities['join_paths']):
                    score += 0.15

            scored_contexts.append({
                'context': ctx,
                'score': score
            })

        # Sort by score
        scored_contexts.sort(key=lambda x: x['score'], reverse=True)

        # Return top contexts (max 5)
        top_contexts = [item['context'] for item in scored_contexts[:5]]

        return top_contexts