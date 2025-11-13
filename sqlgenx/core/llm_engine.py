"""Enhanced LLM engine with natural language schema and auto-fix capabilities."""
from typing import List, Dict, Any, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import hashlib
from dataclasses import dataclass
from rich.markdown import Markdown


@dataclass
class GenerationAttempt:
    """Track SQL generation attempts for circuit breaker."""
    sql: str
    error: Optional[str]
    error_hash: str
    attempt_number: int


class LLMEngineEnhanced:
    """Enhanced LLM engine with natural language schema formatting and auto-fix."""
    
    MAX_RETRY_ATTEMPTS = 2  # Circuit breaker limit
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize enhanced LLM engine."""
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=2048
        )
        
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
    
    def explain_query_as_markdown(self, sql_query: str) -> Markdown:
        """Generate explanation for a SQL query and return as a Rich Markdown object."""
        explanation_text = self.explain_query(sql_query)
        
        # Membersihkan artefak yang tidak diinginkan dari output LLM jika ada
        if "Let's break it down step-by-step:" in explanation_text:
            explanation_text = explanation_text.split("Let's break it down step-by-step:", 1)[1].strip()
            
        return Markdown(explanation_text, style="info")
    
    def generate_sql(
        self,
        user_query: str,
        schema_context: str,  # Natural language formatted schema
        dbms_type: str = "generic",
        auto_fix: bool = False,
        validator = None
    ) -> Tuple[str, List[GenerationAttempt]]:
        """Generate SQL with optional auto-fix and circuit breaker."""
        
        attempts = []
        
        # First attempt
        sql = self._generate_attempt(user_query, schema_context, dbms_type)
        
        # Validate if validator provided
        if validator:
            validation = validator.validate(sql)
            
            if validation.is_valid:
                attempts.append(GenerationAttempt(
                    sql=sql,
                    error=None,
                    error_hash="",
                    attempt_number=1
                ))
                return sql, attempts
            
            # Track failed attempt
            error_msg = "; ".join(validation.errors)
            error_hash = self._hash_error(error_msg)
            
            attempts.append(GenerationAttempt(
                sql=sql,
                error=error_msg,
                error_hash=error_hash,
                attempt_number=1
            ))
            
            # Auto-fix if enabled
            if auto_fix:
                return self._auto_fix_with_circuit_breaker(
                    user_query=user_query,
                    schema_context=schema_context,
                    dbms_type=dbms_type,
                    failed_sql=sql,
                    error_msg=error_msg,
                    error_hash=error_hash,
                    attempts=attempts,
                    validator=validator
                )
        
        # No validator or no auto-fix
        attempts.append(GenerationAttempt(
            sql=sql,
            error=None,
            error_hash="",
            attempt_number=1
        ))
        
        return sql, attempts
    
    def validate_user_query(self, user_query: str, table_names: List[str], schema_context: str = "") -> (bool, str):
        """
        Validates if the user query is relevant to the database schema.
        Returns a tuple of (is_valid, reason).
        """
        if schema_context:
            # Use detailed schema context for more accurate validation
            prompt = f"""
            You are a database analyst. Your task is to determine if a user's question is answerable given detailed schema information.
            The user's question does not need to mention column names directly, but it should be conceptually related to the available data.

            Schema Information:
            {schema_context}

            User query: "{user_query}"

            Is this query relevant and likely answerable using the schema provided?
            Respond with only "VALID" or "INVALID: [detailed reason with specific column references]".

            Example 1:
            Schema: Table 'products' has columns: 'id', 'name', 'price', 'cost'; Table 'orders' has columns: 'id', 'product_id', 'quantity'
            User query: "show me top selling products" 
            Response: VALID

            Example 2:
            Schema: Table 'products' has columns: 'id', 'name', 'selling_price', 'purchase_price'
            User query: "calculate profit margin for each product"
            Response: VALID

            Example 3:
            Schema: Table 'products' has columns: 'id', 'name', 'selling_price' but no cost-related columns
            User query: "calculate profit margin" 
            Response: INVALID: The schema provides selling price information but does not contain explicit cost price information needed to calculate profit margin. Profit margin requires both selling price and cost price.
            """
        else:
            # Fallback to basic table names validation
            prompt = f"""
            You are a database analyst. Your task is to determine if a user's question is answerable given a list of table names.
            The user's question does not need to mention table names directly, but it should be conceptually related to them.

            Available tables: {', '.join(table_names)}

            User query: "{user_query}"

            Is this query relevant and likely answerable using the tables provided?
            Respond with only "VALID" or "INVALID: [brief reason]".

            Example 1:
            Tables: customers, orders, products
            User query: "show me top selling products"
            Response: VALID

            Example 2:
            Tables: employees, departments, salaries
            User query: "what is the weather tomorrow?"
            Response: INVALID: The query is about weather, which is unrelated to employees and departments.
            """
        
        response = self.llm.invoke(prompt).content.strip()
        if response.upper() == "VALID":
            return True, ""
        else:
            reason = response.split(":", 1)[1].strip() if ":" in response else "The query seems unrelated to the database schema."
            return False, reason
        
    def suggest_optimizations(self, sql_query: str, dbms_type: str) -> str:
        """Suggests optimizations for a given SQL query."""
        prompt = f"""You are an expert DBA specializing in {dbms_type}.
        Review the following SQL query for performance bottlenecks and suggest improvements.
        Focus on index recommendations, join efficiency, and filter improvements.
        If the query is already optimal, say so.
        Provide your suggestions in a concise, bulleted list.

        Query:
        ```sql
        {sql_query}
        ```
        """
        response = self.llm.invoke(prompt).content
        return response

    def refine_sql_with_suggestions(self, original_sql: str, user_query: str, suggestions: str, dbms_type: str) -> str:
        """Rewrites an SQL query based on optimization suggestions."""
        prompt = f"""You are an expert SQL engineer tasked with optimizing a query.
        You will be given an original SQL query, the user's request, and a list of optimization suggestions from a DBA.
        Your goal is to rewrite the query to incorporate these suggestions while still fulfilling the user's original request.

        User Request: "{user_query}"

        Original SQL Query:
        ```sql
        {original_sql}
        ```

        DBA's Optimization Suggestions:
        {suggestions}

        Rewrite the SQL query to be more optimal, using {dbms_type} syntax.
        Return ONLY the rewritten SQL query, with no other text or markdown.
        """
        response = self.llm.invoke(prompt).content
        return self._extract_sql(response)

    def create_query_plan(self, user_query: str, schema_context: List[Dict[str, Any]]) -> str:
        """Create a step-by-step plan to answer a complex query."""
        context_text = self._format_schema_context(schema_context)
        prompt = f"""You are a senior data analyst. Your job is to create a clear, step-by-step plan to answer a user's question based on a database schema.

        If the question is simple and can be answered in one go, the plan should be a single step.
        If the question is complex (e.g., requires comparisons, sub-calculations, or multiple steps), break it down into logical parts. Suggest using Common Table Expressions (CTEs) to organize the logic.

        Schema Context:
        {context_text}

        User's Question: "{user_query}"

        Based on the schema and the user's question, provide a concise execution plan.

        Example for a complex query:
        Plan:
        1. Create a CTE to calculate the total revenue for the current month.
        2. Create a second CTE to calculate the average monthly revenue over the last 6 months.
        3. Join these CTEs to compare the current month's revenue against the average.
        """
        response = self.llm.invoke(prompt).content
        return response

    
    def _generate_attempt(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str
    ) -> str:
        """Single SQL generation attempt."""
        
        system_msg = self.system_prompt.format(
            schema_context=schema_context,
            dbms_type=dbms_type
        )
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_query)
        ]
        
        response = self.llm.invoke(messages)
        sql = self._extract_sql(response.content)
        
        return sql
    
    def _auto_fix_with_circuit_breaker(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str,
        failed_sql: str,
        error_msg: str,
        error_hash: str,
        attempts: List[GenerationAttempt],
        validator
    ) -> Tuple[str, List[GenerationAttempt]]:
        """Attempt to fix SQL with circuit breaker to prevent infinite loops."""
        
        seen_errors = {error_hash}
        attempt_num = 2
        
        while attempt_num <= self.MAX_RETRY_ATTEMPTS:
            # Generate fix
            fixed_sql = self._generate_fix(
                failed_sql=failed_sql,
                error_message=error_msg,
                schema_context=schema_context,
                dbms_type=dbms_type
            )
            
            # Validate fix
            validation = validator.validate(fixed_sql)
            
            if validation.is_valid:
                # Success!
                attempts.append(GenerationAttempt(
                    sql=fixed_sql,
                    error=None,
                    error_hash="",
                    attempt_number=attempt_num
                ))
                return fixed_sql, attempts
            
            # Still failing
            new_error_msg = "; ".join(validation.errors)
            new_error_hash = self._hash_error(new_error_msg)
            
            attempts.append(GenerationAttempt(
                sql=fixed_sql,
                error=new_error_msg,
                error_hash=new_error_hash,
                attempt_number=attempt_num
            ))
            
            # Circuit breaker: same error hash = stuck in loop
            if new_error_hash in seen_errors:
                break
            
            seen_errors.add(new_error_hash)
            failed_sql = fixed_sql
            error_msg = new_error_msg
            error_hash = new_error_hash
            attempt_num += 1
        
        # All attempts failed, return last attempt
        return attempts[-1].sql, attempts
    
    def _generate_fix(
        self,
        failed_sql: str,
        error_message: str,
        schema_context: str,
        dbms_type: str
    ) -> str:
        """Generate fixed SQL based on error."""
        
        prompt = self.fix_prompt.format(
            failed_sql=failed_sql,
            error_message=error_message,
            schema_context=schema_context,
            dbms_type=dbms_type
        )
        
        messages = [
            SystemMessage(content="You are an expert SQL debugger."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return self._extract_sql(response.content)
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        sql = response.strip()
        
        # Remove markdown code blocks
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        
        if sql.endswith("```"):
            sql = sql[:-3]
        
        sql = sql.strip()
        
        # Remove any explanatory text before or after SQL
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
        lines = sql.split("\n")
        
        # Find first line with SQL keyword
        start_idx = 0
        for i, line in enumerate(lines):
            if any(line.strip().upper().startswith(kw) for kw in sql_keywords):
                start_idx = i
                break
        
        # Find last line with semicolon or SQL content
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and (line.endswith(";") or any(c.isalnum() for c in line)):
                end_idx = i + 1
                break
        
        sql = "\n".join(lines[start_idx:end_idx]).strip()
        
        return sql
    
    def _hash_error(self, error_msg: str) -> str:
        """Create hash of error message for circuit breaker."""
        return hashlib.md5(error_msg.encode()).hexdigest()[:8]
    
    def explain_query(self, sql_query: str, schema_context: str = "") -> str:
        """Generate explanation for a SQL query."""
        
        prompt = f"""Explain this SQL query in simple terms:

{sql_query}

{f'Schema context: {schema_context}' if schema_context else ''}

Provide a clear explanation covering:
1. What data is being retrieved
2. Which tables are involved and how they're joined
3. Any filtering or aggregation logic
4. The expected result structure"""
        
        messages = [
            SystemMessage(content="You are a SQL educator who explains queries clearly."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content