"""
Optimized LangGraph workflow with reduced API calls.
Replace the existing graph.py with this file.
"""
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path
from langgraph.graph import StateGraph, END

from .schema_loader import SchemaLoader, SchemaInfo
from .vector_store import VectorStore
from .llm_engine import LLMEngine
from .validator import SQLValidator, ValidationResult
from ..utils.output_sanitizer import OutputSanitizer


class GraphState(TypedDict):
    """State for the SQL generation workflow."""
    workspace_dir: Path
    user_query: str
    dbms_type: str
    schema_info: Optional[SchemaInfo]
    retrieved_contexts: List[Dict[str, Any]]
    execution_plan: str
    generated_sql: str
    validation_result: Optional[ValidationResult]
    refinement_count: int
    final_sql: str
    errors: List[str]
    warnings: List[str]
    
    # Optimization control
    skip_optimization: bool
    optimization_suggestions: Optional[str]


class SQLGenerationGraph:
    """Optimized LangGraph workflow for generating SQL queries."""
    
    def __init__(self, api_key: str, workspace_dir: Optional[Path] = None):
        """
        Initialize the workflow graph.
        
        Args:
            api_key: Gemini API key
            workspace_dir: Workspace directory for caching
        """
        self.max_refinements = 1  
        self.api_key = api_key
        self.workspace_dir = workspace_dir
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("load_schema", self._load_schema)
        workflow.add_node("validate_nl_query", self._validate_nl_query)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("create_plan", self._create_plan)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql", self._validate_sql)
        workflow.add_node("optimize_if_needed", self._optimize_if_needed)
        
        # Define edges
        workflow.set_entry_point("load_schema")
        workflow.add_edge("load_schema", "validate_nl_query")
        
        workflow.add_conditional_edges(
            "validate_nl_query",
            self._decide_after_nl_validation,
            {"continue": "retrieve_context", "end": END}
        )
        
        workflow.add_edge("retrieve_context", "create_plan")
        workflow.add_edge("create_plan", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        
        workflow.add_conditional_edges(
            "validate_sql",
            self._decide_after_validation,
            {"optimize": "optimize_if_needed", "end": END}
        )
        
        workflow.add_conditional_edges(
            "optimize_if_needed",
            self._decide_after_optimization,
            {"refine": "validate_sql", "end": END}
        )
        
        return workflow.compile()
    
    def _load_schema(self, state: GraphState) -> GraphState:
        """Load schema from workspace."""
        workspace_dir = state["workspace_dir"]
        schema_file = workspace_dir / "schema.sql"
        
        if not schema_file.exists():
            state["errors"].append("Schema file not found in workspace")
            return state
        
        try:
            dialect = state.get("dbms_type", "")
            if dialect == "generic":
                dialect = ""
            
            loader = SchemaLoader(dialect=dialect)
            schema_info = loader.load_from_file(schema_file)
            
            if not schema_info or not schema_info.tables:
                state["errors"].append("No tables found in schema")
                return state
            
            state["schema_info"] = schema_info
        except Exception as e:
            state["errors"].append(f"Failed to load schema: {str(e)}")
        
        return state
    
    def _validate_nl_query(self, state: GraphState) -> GraphState:
        """Validate the user's natural language query."""
        if state.get("errors") or not state["schema_info"]:
            return state
        
        try:
            llm_engine = LLMEngine(self.api_key, self.workspace_dir)
            
            # Build detailed schema context
            detailed_schema = []
            for table in state["schema_info"].tables:
                cols = ", ".join([f"'{c.name}' ({c.data_type})" for c in table.columns])
                detailed_schema.append(f"Table '{table.name}': {cols}")
            
            schema_context = " | ".join(detailed_schema)
            table_names = [t.name for t in state["schema_info"].tables]
            
            is_valid, reason = llm_engine.validate_user_query(
                state["user_query"],
                table_names,
                schema_context
            )
            
            if not is_valid:
                state["errors"].append(f"Query validation failed: {reason}")
        
        except Exception as e:
            state["errors"].append(f"NL validation error: {str(e)}")
        
        return state
    
    def _decide_after_nl_validation(self, state: GraphState) -> str:
        """Decide whether to continue after NL validation."""
        return "end" if state.get("errors") else "continue"
    
    def _retrieve_context(self, state: GraphState) -> GraphState:
        """Retrieve relevant schema context."""
        if state.get("errors"):
            return state
        
        try:
            vector_store = VectorStore(state["workspace_dir"])
            contexts = vector_store.retrieve_context(state["user_query"], n_results=5)
            state["retrieved_contexts"] = contexts
        except Exception as e:
            state["errors"].append(f"Context retrieval failed: {str(e)}")
        
        return state
    
    def _create_plan(self, state: GraphState) -> GraphState:
        """Create execution plan based on user query and schema context."""
        if state.get("errors"):
            return state

        try:
            llm_engine = LLMEngine(self.api_key, self.workspace_dir)

            # Format schema context
            schema_context = "\n".join([
                ctx.get("content", "") for ctx in state["retrieved_contexts"]
            ])

            # Create execution plan (with caching support)
            execution_plan = llm_engine.create_execution_plan(
                state["user_query"],
                schema_context,
                state.get("dbms_type", "generic")
            )

            # Check if the plan indicates validation failure
            if execution_plan.startswith("VALIDATION_FAILED:"):
                state["errors"].append(f"Query validation failed: {execution_plan[18:].strip()}")
                return state

            # Clean output (sanitizer already applied in LLM engine)
            state["execution_plan"] = execution_plan

        except Exception as e:
            state["errors"].append(f"Plan creation failed: {str(e)}")

        return state

    def _generate_sql(self, state: GraphState) -> GraphState:
        """Generate SQL based on execution plan."""
        if state.get("errors") or not state.get("execution_plan"):
            return state

        try:
            llm_engine = LLMEngine(self.api_key, self.workspace_dir)

            # Format schema context
            schema_context = "\n".join([
                ctx.get("content", "") for ctx in state["retrieved_contexts"]
            ])

            # Generate SQL based on the plan (with caching support)
            sql_query = llm_engine.generate_sql_from_plan(
                state["user_query"],
                state["execution_plan"],
                schema_context,
                state.get("dbms_type", "generic")
            )

            # Clean output (sanitizer already applied in LLM engine)
            state["generated_sql"] = sql_query

        except Exception as e:
            state["errors"].append(f"SQL generation failed: {str(e)}")

        return state
    
    def _validate_sql(self, state: GraphState) -> GraphState:
        """Validate generated SQL."""
        if state.get("errors"):
            return state
        
        try:
            validator = SQLValidator(dialect=state.get("dbms_type", ""))
            validation = validator.validate(state["generated_sql"])
            state["validation_result"] = validation
            
            if validation.is_valid:
                formatted_sql = validator.format_sql(state["generated_sql"])
                
                # Check schema compatibility
                if state["schema_info"]:
                    available_tables = [t.name for t in state["schema_info"].tables]
                    is_compat, missing = validator.check_schema_compatibility(
                        state["generated_sql"],
                        available_tables
                    )
                    
                    if not is_compat:
                        state["errors"].append(
                            f"Query uses tables not in schema: {', '.join(missing)}"
                        )
                    else:
                        # Normalize table name case
                        import re
                        table_case_map = {t.lower(): t for t in available_tables}
                        
                        for lower_name, actual_name in table_case_map.items():
                            pattern = r'\b(' + re.escape(lower_name) + r')\b'
                            formatted_sql = re.sub(
                                pattern, 
                                actual_name, 
                                formatted_sql, 
                                flags=re.IGNORECASE
                            )
                
                state["final_sql"] = formatted_sql
            else:
                state["errors"].extend(validation.errors)
        
        except Exception as e:
            state["errors"].append(f"Validation failed: {str(e)}")
        
        return state
    
    def _decide_after_validation(self, state: GraphState) -> str:
        """Decide whether to optimize after validation."""
        # If errors, end immediately
        if state.get("errors"):
            return "end"
        
        # If no SQL generated, end
        if not state.get("final_sql"):
            return "end"
        
        # Check if we should skip optimization
        llm_engine = LLMEngine(self.api_key, self.workspace_dir)
        
        # ✅ KEY OPTIMIZATION: Skip optimization for simple queries
        if not llm_engine.should_optimize(state["final_sql"]):
            state["skip_optimization"] = True
            state["warnings"].append("Query is simple, skipping optimization")
            return "end"
        
        # Check refinement limit
        if state.get("refinement_count", 0) >= self.max_refinements:
            return "end"
        
        return "optimize"
    
    def _optimize_if_needed(self, state: GraphState) -> GraphState:
        """Suggest optimizations and refine SQL if needed."""
        if state.get("errors") or state.get("skip_optimization"):
            return state
        
        try:
            llm_engine = LLMEngine(self.api_key, self.workspace_dir)
            
            # Get optimization suggestions (with caching)
            suggestions = llm_engine.suggest_optimizations(
                state["final_sql"],
                state["dbms_type"]
            )
            
            state["optimization_suggestions"] = suggestions
            
            # Check if optimization is actually needed
            if "already" in suggestions.lower() and "optimal" in suggestions.lower():
                state["warnings"].append("Query is already well-optimized")
                return state
            
            # Refine SQL based on suggestions
            refined_sql = llm_engine.refine_sql_with_suggestions(
                original_sql=state["final_sql"],
                user_query=state["user_query"],
                suggestions=suggestions,
                dbms_type=state["dbms_type"]
            )
            
            # Update state
            state["generated_sql"] = refined_sql
            state["refinement_count"] = state.get("refinement_count", 0) + 1
            
        except Exception as e:
            state["errors"].append(f"Optimization failed: {str(e)}")
        
        return state
    
    def _decide_after_optimization(self, state: GraphState) -> str:
        """Decide whether to continue refining."""
        refinement_count = state.get("refinement_count", 0)
        suggestions = state.get("optimization_suggestions", "")
        
        # Reached refinement limit
        if refinement_count >= self.max_refinements:
            return "end"
        
        # No meaningful suggestions
        if not suggestions or "already optimal" in suggestions.lower():
            return "end"
        
        # Continue refining
        return "refine"
    
    def run(
        self,
        workspace_dir: Path,
        user_query: str,
        dbms_type: str = "generic"
    ) -> GraphState:
        """
        Run the SQL generation workflow.
        
        Args:
            workspace_dir: Workspace directory
            user_query: User's natural language query
            dbms_type: Database type
            
        Returns:
            Final graph state
        """
        initial_state: GraphState = {
            "workspace_dir": workspace_dir,
            "user_query": user_query,
            "dbms_type": dbms_type,
            "schema_info": None,
            "retrieved_contexts": [],
            "execution_plan": "",
            "generated_sql": "",
            "validation_result": None,
            "refinement_count": 0,
            "final_sql": "",
            "errors": [],
            "warnings": [],
            "skip_optimization": False,
            "optimization_suggestions": None
        }
        
        result = self.graph.invoke(initial_state)
        return result