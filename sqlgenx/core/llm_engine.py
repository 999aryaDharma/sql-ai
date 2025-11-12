"""LLM engine using LangChain and Google Gemini."""
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from rich.markdown import Markdown


class LLMEngine:
    """Manages LLM interactions for SQL generation."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize LLM engine with Gemini."""
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1,  # Low temperature for more deterministic SQL
            max_tokens=10000
        )
        
        self.system_prompt = """You are an expert SQL engineer.
You are given a database schema and must generate syntactically correct SQL queries.

CRITICAL RULES:
1. Only use tables and columns that exist in the provided schema context
2. Return ONLY the SQL query, nothing else - no explanations, no markdown, no code blocks
3. Use proper SQL syntax for the database type specified
4. Include appropriate JOINs when querying multiple tables
5. Use meaningful aliases for better readability
6. For aggregations, always include GROUP BY when needed
7. Pay close attention to sample values provided in comments (e.g., `status ('Sample values: 'completed', 'pending')`) to correctly filter data.
8. If a step-by-step plan is provided, follow it to construct the query, using CTEs (Common Table Expressions) for multi-step logic.
9. Use LIMIT for queries that might return many rows

Schema Context:
{schema_context}

Generate a SQL query for the following request:"""
    
    def generate_sql(
        self, 
        user_query: str, 
        plan: str,
        schema_context: List[Dict[str, Any]],
        dbms_type: str = "generic"
    ) -> str:
        """Generate SQL query from natural language."""
        
        # Format schema context
        context_text = self._format_schema_context(schema_context)
        
        # Create prompt
        system_msg = self.system_prompt.format(schema_context=context_text)
        
        # Add DBMS-specific instructions
        if dbms_type.lower() == "postgresql":
            system_msg += "\n\nUse PostgreSQL syntax (e.g., use double quotes for identifiers if needed)."
        elif dbms_type.lower() == "mysql":
            system_msg += "\n\nUse MySQL syntax (e.g., backticks for identifiers if needed)."
        elif dbms_type.lower() == "sqlite":
            system_msg += "\n\nUse SQLite syntax."
        
        # Construct final prompt with plan and a firm reminder
        final_user_prompt = f"""User Request: {user_query}

Execution Plan:
{plan}

CRITICAL REMINDER: Based on the plan above, provide ONLY the final, complete SQL query. Do not add any explanation or markdown."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=final_user_prompt)
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Extract SQL from response
        sql = self._extract_sql(response.content)
        
        return sql
    
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
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Remove markdown code blocks if present
        sql = response.strip()
        
        # Remove ```sql and ``` markers
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        
        if sql.endswith("```"):
            sql = sql[:-3]
        
        sql = sql.strip()
        
        # Remove any explanatory text before or after the SQL
        # Look for common SQL keywords at the start
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
    
    def explain_query(self, sql_query: str) -> str:
        """Generate explanation for a SQL query."""
        messages = [
            SystemMessage(content="You are a SQL expert. Explain the following SQL query in simple terms."),
            HumanMessage(content=f"Jelaskan SQL query ini dengan bahasa indonesia:\n\n{sql_query}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def explain_query_as_markdown(self, sql_query: str) -> Markdown:
        """Generate explanation for a SQL query and return as a Rich Markdown object."""
        explanation_text = self.explain_query(sql_query)
        
        # Membersihkan artefak yang tidak diinginkan dari output LLM jika ada
        if "Let's break it down step-by-step:" in explanation_text:
            explanation_text = explanation_text.split("Let's break it down step-by-step:", 1)[1].strip()
            
        return Markdown(explanation_text, style="info")

    def validate_user_query(self, user_query: str, table_names: List[str]) -> (bool, str):
        """
        Validates if the user query is relevant to the database schema.
        Returns a tuple of (is_valid, reason).
        """
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