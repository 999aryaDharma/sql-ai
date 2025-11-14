"""
Updated LLM Engine with integrated cache, rate limiting, and output sanitization.
Replace the existing llm_engine.py with this file.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import hashlib
from rich.markdown import Markdown

# Import new modules
from .cache.manager import CacheManager
from .rate_limiting.limiter import global_rate_limiter, global_circuit_breaker
from ..utils.output_sanitizer import OutputSanitizer


class LLMEngine:
    """Enhanced LLM engine with caching, rate limiting, and clean output."""

    def __init__(
        self,
        api_key: str,
        workspace_dir: Optional[Path] = None,
        model: str = "gemini-2.5-flash-lite",
        enable_cache: bool = True
    ):
        """
        Initialize LLM engine.

        Args:
            api_key: Gemini API key
            workspace_dir: Optional workspace directory for caching
            model: Model name
            enable_cache: Enable caching (default: True)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=50000
        )

        # Initialize cache if workspace provided
        self.cache = None
        if enable_cache and workspace_dir:
            self.cache = CacheManager(workspace_dir)

        # Use global rate limiter and circuit breaker
        self.rate_limiter = global_rate_limiter
        self.circuit_breaker = global_circuit_breaker

        self.system_prompt = """You are an expert SQL engineer with deep understanding of database schemas and relationships.

CRITICAL RULES:
1. Only use tables and columns that exist in the provided schema
2. Return ONLY the SQL query - no explanations, no markdown, no code blocks
3. Use proper SQL syntax for the specified database type
4. Follow the relationship hints to construct proper JOINs
5. Use meaningful table aliases for readability
6. For aggregations, always include GROUP BY when needed
7. Use LIMIT for queries that might return many rows

When you see relationship hints like "customers are related to products through orders",
this means you need to JOIN: customers → orders → products

Schema Context (in natural language):
{schema_context}

Database Type: {dbms_type}

Generate a SQL query for the following request:"""

        self.fix_prompt = """The previous SQL query failed with an error.

Original Query:
{failed_sql}

Error Message:
{error_message}

Schema Context:
{schema_context}

Please fix the SQL query. Consider:
1. Check if all table and column names exist in the schema
2. Verify JOIN conditions are correct
3. Ensure proper syntax for the database type: {dbms_type}
4. Check for missing GROUP BY clauses with aggregations

Return ONLY the corrected SQL query, nothing else."""

    def should_optimize(self, sql_query: str) -> bool:
        """
        Quick heuristic check if SQL needs optimization.

        Args:
            sql_query: SQL query to check

        Returns:
            True if optimization might be beneficial
        """
        sql_upper = sql_query.upper()

        # Already has LIMIT
        has_limit = 'LIMIT' in sql_upper

        # Uses SELECT *
        has_select_star = 'SELECT *' in sql_upper or 'SELECT\n*' in sql_upper

        # Has CTEs (already optimized pattern)
        has_cte = 'WITH' in sql_upper

        # Simple query (no joins, single table)
        is_simple = sql_upper.count('JOIN') == 0 and sql_upper.count('FROM') == 1

        # Query length (very short queries likely don't need optimization)
        is_short = len(sql_query) < 100

        # Skip optimization if query is already good
        if has_limit and not has_select_star and (has_cte or is_simple or is_short):
            return False

        return True

    def _make_api_call(self, messages: List) -> str:
        """
        Make API call with rate limiting.

        Args:
            messages: Messages to send to LLM

        Returns:
            LLM response content
        """
        # Rate limit check
        wait_time = self.rate_limiter.wait_if_needed()

        # Make API call
        response = self.llm.invoke(messages)

        return response.content

    def _format_schema_context(self, contexts: List[Dict[str, Any]]) -> str:
        """Format retrieved schema contexts for prompt."""
        if not contexts:
            return "No schema context available."

        formatted = []
        for ctx in contexts:
            table = ctx.get("table", "unknown")
            content = ctx.get("content", "")
            formatted.append(f"=== {table} ===\n{content}")

        return "\n\n".join(formatted)

    def generate_sql(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str = "generic"
    ) -> str:
        """
        Generate SQL with caching support.

        Args:
            user_query: User's natural language query
            schema_context: Schema context string
            dbms_type: Database type

        Returns:
            Generated SQL query
        """
        # Check cache first
        if self.cache:
            cached_sql = self.cache.get_sql_generation(
                user_query=user_query,
                schema_context=schema_context,
                dbms_type=dbms_type
            )

            if cached_sql:
                return cached_sql

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt(user_query):
            raise RuntimeError(
                "Circuit breaker is open. Too many failed attempts. "
                "Please try a different query or wait a few minutes."
            )

        # Generate SQL
        try:
            system_msg = self.system_prompt.format(
                schema_context=schema_context,
                dbms_type=dbms_type
            )

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_query)
            ]

            response_content = self._make_api_call(messages)

            # Clean output
            sql = OutputSanitizer.clean_sql(response_content)

            # Cache result
            if self.cache and sql:
                self.cache.set_sql_generation(
                    user_query=user_query,
                    schema_context=schema_context,
                    dbms_type=dbms_type,
                    sql=sql
                )

            # Record success
            self.circuit_breaker.record_success(user_query)

            return sql

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_attempt(user_query, str(e))
            raise

    def validate_user_query(
        self,
        user_query: str,
        table_names: List[str],
        schema_context: str = ""
    ) -> Tuple[bool, str]:
        """
        Validate if user query is relevant to schema.

        Args:
            user_query: User's natural language query
            table_names: Available table names
            schema_context: Detailed schema context

        Returns:
            Tuple of (is_valid, reason)
        """
        if schema_context:
            prompt = f"""You are a database analyst. Determine if a user's question is answerable given detailed schema information.

Schema Information:
{schema_context}

User query: "{user_query}"

CRITICAL: When evaluating, consider:
1. Can required data be accessed through direct table access?
2. Can required data be accessed through table relationships (joins)?
3. For temporal queries: Can dates be accessed through related tables if not in the primary table?
4. For example: If user wants date-based filtering on items that don't have date columns, but the items are linked to transactions that have dates, this IS POSSIBLE through JOIN.
5. Only mark as invalid if there is truly NO PATH to access required information.

Is this query relevant and answerable using the schema provided (including through table relationships)?
Respond with only "VALID" or "INVALID: [reason]"."""
        else:
            prompt = f"""You are a database analyst. Determine if a user's question is answerable given table names.

Available tables: {', '.join(table_names)}

User query: "{user_query}"

Is this query relevant and answerable using the tables provided?
Respond with only "VALID" or "INVALID: [reason]"."""

        response = self._make_api_call([HumanMessage(content=prompt)])

        if response.upper().startswith("VALID"):
            return True, ""
        else:
            reason = response.split(":", 1)[1].strip() if ":" in response else "Query unrelated to schema"
            return False, reason

    def suggest_optimizations(self, sql_query: str, dbms_type: str) -> str:
        """
        Suggest optimizations for SQL query (with caching).

        Args:
            sql_query: SQL query to optimize
            dbms_type: Database type

        Returns:
            Optimization suggestions
        """
        # Check cache
        if self.cache:
            cached = self.cache.get_optimization(sql_query, dbms_type)
            if cached:
                return cached

        prompt = f"""You are an expert DBA specializing in {dbms_type}.
Review the following SQL query for performance improvements.
Focus on index recommendations, join efficiency, and filter improvements.
If the query is already optimal, say "Query is already well-optimized."
Provide concise, bulleted suggestions.

Query:
{sql_query}"""

        response = self._make_api_call([HumanMessage(content=prompt)])
        suggestions = OutputSanitizer.clean_text(response)

        # Cache result
        if self.cache:
            self.cache.set_optimization(sql_query, dbms_type, suggestions)

        return suggestions

    def refine_sql_with_suggestions(
        self,
        original_sql: str,
        user_query: str,
        suggestions: str,
        dbms_type: str
    ) -> str:
        """
        Refine SQL based on optimization suggestions.

        Args:
            original_sql: Original SQL query
            user_query: User's original request
            suggestions: Optimization suggestions
            dbms_type: Database type

        Returns:
            Refined SQL query
        """
        prompt = f"""You are an expert SQL engineer optimizing a query.
Rewrite the query to incorporate these optimization suggestions while maintaining correctness.

User Request: "{user_query}"

Original SQL:
{original_sql}

Optimization Suggestions:
{suggestions}

Return ONLY the optimized SQL query for {dbms_type}, with no other text."""

        response = self._make_api_call([HumanMessage(content=prompt)])

        return OutputSanitizer.clean_sql(response)
        
    def create_execution_plan(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str
    ) -> str:
        """
        Create execution plan based on user query and schema context.

        Args:
            user_query: User's natural language query
            schema_context: Schema context string
            dbms_type: Database type

        Returns:
            Execution plan string
        """
        prompt = f"""You are an expert SQL engineer with deep understanding of database schemas.

Before creating the execution plan, carefully analyze the schema and identify both available capabilities and potential limitations.

Schema Context:
{schema_context}

Database Type: {dbms_type}

User Request: "{user_query}"

CRITICAL: Before creating the plan, do these checks:
1. Identify all required entities (tables, columns, date fields, etc.) from the user request
2. Verify if these entities exist in the schema directly (in the same table) OR if they can be accessed through relationships/joins
3. Check for date filtering capabilities - can required dates be accessed through joins if not directly available?
4. Map out table relationships - which tables can be joined to get required data?
5. Pay special attention to temporal columns and how they can be accessed through joins
6. Only mark as impossible if there's truly NO PATH to access required data

For example: If transaction_item doesn't have a date column but is linked to a transaction table that has a date, this is possible through JOIN.
Do NOT assume missing functionality if it can be achieved through table relationships.

If the data truly cannot be accessed due to schema limitations, return "VALIDATION_FAILED: [specific explanation of what's missing and why no alternative path exists]"

If the query is possible, then create a detailed execution plan considering:
1. Tables involved and their relationships
2. Required joins and their conditions (be explicit about how to connect tables)
3. Filters and where clauses (including date filters that may come from related tables)
4. Aggregations and groupings needed
5. Any date range considerations (including how to access dates through joins)
6. How to properly group/aggregate across related tables

Provide only the execution plan if possible, or the validation failure message:"""

        response = self._make_api_call([HumanMessage(content=prompt)])
        execution_plan = OutputSanitizer.clean_text(response)
        return execution_plan

    def generate_sql_from_plan(
        self,
        user_query: str,
        execution_plan: str,
        schema_context: str,
        dbms_type: str
    ) -> str:
        """
        Generate SQL based on execution plan and schema context.

        Args:
            user_query: User's natural language query
            execution_plan: Execution plan to follow
            schema_context: Schema context string
            dbms_type: Database type

        Returns:
            Generated SQL query
        """
        prompt = f"""You are an expert SQL engineer with deep understanding of database schemas.

Generate SQL query following this execution plan:
{execution_plan}

Schema Context:
{schema_context}

Database Type: {dbms_type}

User Request: "{user_query}"

CRITICAL RULES:
1. Only use tables and columns that exist in the provided schema
2. Follow the execution plan exactly, paying special attention to specified joins and relationships
3. For temporal data: if requested date filtering requires joining to other tables, ensure those joins are included
4. Return ONLY the SQL query - no explanations, no markdown, no code blocks
5. Use proper {dbms_type} syntax
6. Ensure all table and column names are correct per the schema
7. When joining tables, use appropriate aliases and qualify column names to prevent ambiguity
8. For complex queries, verify that all required joins and filters are included
9. If the plan indicates joining tables to access temporal data, make sure the JOINs and date filters are properly implemented"""

        response = self._make_api_call([HumanMessage(content=prompt)])
        sql_query = OutputSanitizer.clean_sql(response)
        return sql_query

    def create_plan_and_generate_sql(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str
    ) -> Tuple[str, str]:
        """
        Create execution plan and generate SQL in single call.

        Args:
            user_query: User's natural language query
            schema_context: Schema context string
            dbms_type: Database type

        Returns:
            Tuple of (execution_plan, sql_query)
        """
        prompt = f"""You are an expert SQL engineer with deep understanding of database schemas.

CRITICAL: Return both plan and SQL in this exact format:

PLAN:
[Your execution plan]

SQL:
[Your SQL query]

Schema Context:
{schema_context}

Database Type: {dbms_type}

User Request: "{user_query}"

Rules:
1. Only use tables/columns from the schema
2. Use proper {dbms_type} syntax
3. For complex queries, explain the plan step-by-step
4. For simple queries, plan can be brief
5. SQL must be production-ready"""

        response = self._make_api_call([HumanMessage(content=prompt)])

        # Parse response
        execution_plan = ""
        sql_query = ""

        if "PLAN:" in response and "SQL:" in response:
            parts = response.split("SQL:", 1)
            plan_part = parts[0].replace("PLAN:", "").strip()
            sql_part = parts[1].strip()

            execution_plan = OutputSanitizer.clean_text(plan_part)
            sql_query = OutputSanitizer.clean_sql(sql_part)
        else:
            # Fallback: try to extract SQL
            sql_query = OutputSanitizer.clean_sql(response)
            execution_plan = "Direct SQL generation (no plan provided)"

        return execution_plan, sql_query

    def generate_plan_refine_and_optimize(
        self,
        user_query: str,
        sql_query: str,
        schema_context: List[Dict[str, Any]],
        dbms_type: str
    ) -> Tuple[str, str]:
        """
        Generate plan and optimization suggestions in single call.

        Args:
            user_query: User's query
            sql_query: Current SQL
            schema_context: Schema context
            dbms_type: Database type

        Returns:
            Tuple of (plan, optimizations)
        """
        context_text = self._format_schema_context(schema_context)

        prompt = f"""You are a senior data analyst and DBA. Provide:

1. Execution plan for the user's query
2. Optimization suggestions for the SQL

Schema Context:
{context_text}

User's Question: "{user_query}"

SQL Query:
{sql_query}

Respond in this format:

PLAN:
[execution plan]

OPTIMIZATIONS:
[optimization suggestions or "Query is already well-optimized."]"""

        response = self._make_api_call([HumanMessage(content=prompt)])

        # Parse response
        if "PLAN:" in response and "OPTIMIZATIONS:" in response:
            parts = response.split("OPTIMIZATIONS:", 1)
            plan = parts[0].replace("PLAN:", "").strip()
            optimizations = parts[1].strip()

            return (
                OutputSanitizer.clean_text(plan),
                OutputSanitizer.clean_text(optimizations)
            )

        return response, "Unable to parse response"

    def explain_query(self, sql_query: str, schema_context: str = "") -> str:
        """
        Generate explanation for SQL query.

        Args:
            sql_query: SQL query to explain
            schema_context: Optional schema context

        Returns:
            Explanation text
        """
        prompt = f"""Explain this SQL query in simple terms:

{sql_query}

{f'Schema context: {schema_context}' if schema_context else ''}

Provide a clear explanation covering:
1. What data is being retrieved
2. Which tables are involved and how they're joined
3. Any filtering or aggregation logic
4. The expected result structure"""

        response = self._make_api_call([
            SystemMessage(content="You are a SQL educator who explains queries clearly."),
            HumanMessage(content=prompt)
        ])

        return OutputSanitizer.clean_explanation(response)

    def explain_query_as_markdown(self, sql_query: str) -> Markdown:
        """Get query explanation as Rich Markdown object."""
        explanation = self.explain_query(sql_query)
        return Markdown(explanation, style="info")