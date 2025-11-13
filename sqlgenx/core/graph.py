"""LangGraph workflow for SQL generation."""
from typing import TypedDict, List, Dict, Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from .schema_loader import SchemaLoader, SchemaInfo
from .vector_store import VectorStore
from .llm_engine import LLMEngineEnhanced as LLMEngine
from .validator import SQLValidator, ValidationResult


class GraphState(TypedDict):
    """State for the SQL generation workflow."""
    workspace_dir: Path
    user_query: str
    dbms_type: str
    schema_info: SchemaInfo | None
    retrieved_contexts: List[Dict[str, Any]]
    plan: str | None
    generated_sql: str
    validation_result: ValidationResult | None
    refinement_count: int
    optimizations: str | None
    final_sql: str
    errors: List[str]


class SQLGenerationGraph:
    """LangGraph workflow for generating SQL queries."""
    
    def __init__(self, api_key: str):
        """Initialize the workflow graph."""
        self.max_refinements = 2  # Batasi perulangan untuk menghindari loop tak terbatas
        self.api_key = api_key
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("load_schema", self._load_schema)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("create_plan", self._create_plan)
        workflow.add_node("validate_nl_query", self._validate_nl_query)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql", self._validate_sql)
        workflow.add_node("refine_sql", self._refine_sql)
        workflow.add_node("suggest_optimizations", self._suggest_optimizations)
        
        # Define edges
        workflow.set_entry_point("load_schema")
        workflow.add_edge("load_schema", "validate_nl_query")
        workflow.add_conditional_edges(
            "validate_nl_query",
            self._decide_to_generate,
            {"continue": "retrieve_context", "end": END},
        )
        workflow.add_edge("retrieve_context", "create_plan")
        workflow.add_edge("create_plan", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_conditional_edges(
            "validate_sql",
            lambda s: "suggest_optimizations" if not s.get("errors") else END
        )
        workflow.add_conditional_edges(
            "suggest_optimizations", self._decide_to_refine, {"refine": "refine_sql", "end": END}
        )
        workflow.add_edge("refine_sql", "validate_sql") # <-- Ini adalah siklus rekursif
        
        return workflow.compile()
    
    def _load_schema(self, state: GraphState) -> GraphState:
        """Load schema from workspace."""
        workspace_dir = state["workspace_dir"]
        schema_file = workspace_dir / "schema.sql"
        
        if not schema_file.exists():
            state["errors"].append("Schema file not found in workspace")
            return state
        
        try:
            # Determine dialect
            dialect = state.get("dbms_type", "")
            if dialect == "generic":
                dialect = ""
            
            loader = SchemaLoader(dialect=dialect)
            schema_info = loader.load_from_file(schema_file)
            
            # Verify schema has tables
            if not schema_info or not schema_info.tables:
                state["errors"].append("No tables found in schema")
                return state
            
            state["schema_info"] = schema_info
        except Exception as e:
            state["errors"].append(f"Failed to load schema: {str(e)}")
            import traceback
            print(f"DEBUG - Schema load error: {traceback.format_exc()}")
        
        return state
    
    def _validate_nl_query(self, state: GraphState) -> GraphState:
        """Validate the user's natural language query against the schema."""
        if state.get("errors") or not state["schema_info"]:
            return state
        
        try:
            llm_engine = LLMEngine(self.api_key)
            # Create more detailed schema context including table and column names
            detailed_schema_context = []
            for table in state["schema_info"].tables:
                table_info = f"Table '{table.name}' has columns: "
                column_info = ", ".join([f"'{col.name}' ({col.data_type})" for col in table.columns])
                table_info += column_info
                detailed_schema_context.append(table_info)
                
            schema_context_str = " | ".join(detailed_schema_context)
            
            # Get table names as fallback
            table_names = [t.name for t in state["schema_info"].tables]
            
            # Use the detailed schema context for validation
            is_valid, reason = llm_engine.validate_user_query(
                state["user_query"], 
                table_names, 
                schema_context_str
            )
            
            if not is_valid:
                state["errors"].append(f"Query validation failed: {reason}")
        except Exception as e:
            state["errors"].append(f"Error during NL query validation: {str(e)}")
            
        return state

    def _decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to continue generation or end."""
        if state.get("errors"):
            return "end"
        return "continue"

    def _retrieve_context(self, state: GraphState) -> GraphState:
        """Retrieve relevant schema context from vector store."""
        if state.get("errors"):
            return state
        
        try:
            vector_store = VectorStore(state["workspace_dir"])
            contexts = vector_store.retrieve_context(state["user_query"], n_results=5)
            state["retrieved_contexts"] = contexts
        except Exception as e:
            state["errors"].append(f"Failed to retrieve context: {str(e)}")
        
        return state
    
    def _create_plan(self, state: GraphState) -> GraphState:
        """Create a query plan using the LLM."""
        if state.get("errors"):
            return state
        
        try:
            llm_engine = LLMEngine(self.api_key)
            plan = llm_engine.create_query_plan(
                state["user_query"],
                state["retrieved_contexts"]
            )
            state["plan"] = plan
        except Exception as e:
            state["errors"].append(f"Failed to create a plan: {str(e)}")
        return state

    def _generate_sql(self, state: GraphState) -> GraphState:
        """Generate SQL using LLM."""
        if state.get("errors"):
            return state
        
        try:
            llm_engine = LLMEngine(self.api_key)
            sql, attempts = llm_engine.generate_sql(
                state["user_query"],
                state["plan"],
                state["retrieved_contexts"],
                state.get("dbms_type", "generic")
            )
            state["generated_sql"] = sql
        except Exception as e:
            state["errors"].append(f"Failed to generate SQL: {str(e)}")
        
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
                # Format the SQL for better readability
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
                        # If compatible, normalize table names to match actual table names in database
                        # Create mapping from lowercase to actual case
                        table_case_map = {t.lower(): t for t in available_tables}
                        
                        # Replace table names in the SQL to match actual database case
                        import re
                        sql_to_use = formatted_sql
                        for lower_name, actual_name in table_case_map.items():
                            # Replace table names while being careful not to replace partial matches
                            pattern = r'\b(' + re.escape(lower_name) + r')\b'
                            sql_to_use = re.sub(pattern, actual_name, sql_to_use, flags=re.IGNORECASE)
                        
                        formatted_sql = sql_to_use

                state["final_sql"] = formatted_sql
            else:
                state["errors"].extend(validation.errors)
        
        except Exception as e:
            state["errors"].append(f"Failed to validate SQL: {str(e)}")
        
        return state
    
    def _decide_to_refine(self, state: GraphState) -> str:
        """Decide whether to refine the query or end the process."""
        refinement_count = state.get("refinement_count", 0)
        optimizations = state.get("optimizations", "")

        # Jika sudah mencapai batas, berhenti
        if refinement_count >= self.max_refinements:
            state["optimizations"] = None # Hapus sisa saran agar tidak ditampilkan
            return "end"

        # Jika tidak ada saran optimisasi yang berarti, berhenti
        if not optimizations or "already optimal" in optimizations.lower():
            # Ini adalah kondisi keluar yang ideal. Kueri sudah optimal.
            state["optimizations"] = None # Hapus pesan "already optimal"
            return "end"

        # Jika ada saran, lanjutkan untuk perbaikan
        return "refine"

    def _refine_sql(self, state: GraphState) -> GraphState:
        """Refine the SQL query based on optimization suggestions."""
        if state.get("errors"):
            return state

        state["refinement_count"] = state.get("refinement_count", 0) + 1
        
        try:
            llm_engine = LLMEngine(self.api_key)
            refined_sql = llm_engine.refine_sql_with_suggestions(
                original_sql=state["final_sql"],
                user_query=state["user_query"],
                suggestions=state["optimizations"],
                dbms_type=state["dbms_type"]
            )
            state["generated_sql"] = refined_sql # Ganti SQL yang akan divalidasi selanjutnya
        except Exception as e:
            state["errors"].append(f"Failed to refine SQL: {str(e)}")
        return state

    def _suggest_optimizations(self, state: GraphState) -> GraphState:
        """Suggest optimizations for the generated SQL."""
        if state.get("errors") or not state.get("final_sql"):
            return state
        
        try:
            llm_engine = LLMEngine(self.api_key)
            suggestions = llm_engine.suggest_optimizations(
                state["final_sql"],
                state["dbms_type"]
            )
            state["optimizations"] = suggestions
        except Exception as e:
            state["errors"].append(f"Failed to get optimizations: {str(e)}")
        return state
    
    def run(
        self, 
        workspace_dir: Path, 
        user_query: str, 
        dbms_type: str = "generic"
    ) -> GraphState:
        """Run the SQL generation workflow."""
        initial_state: GraphState = {
            "workspace_dir": workspace_dir,
            "user_query": user_query,
            "dbms_type": dbms_type,
            "schema_info": None,
            "retrieved_contexts": [],
            "plan": None,
            "generated_sql": "",
            "validation_result": None,
            "refinement_count": 0,
            "optimizations": None,
            "final_sql": "",
            "errors": []
        }
        
        result = self.graph.invoke(initial_state)
        return result