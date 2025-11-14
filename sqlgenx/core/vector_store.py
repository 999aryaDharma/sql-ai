"""
sqlgenx/core/vector_store.py - UPDATED

Vector store integration with semantic enrichment support.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import json


class VectorStore:
    """Manages vector embeddings for schema context retrieval."""
    
    def __init__(self, workspace_dir: Path):
        """Initialize vector store for a workspace."""
        self.workspace_dir = workspace_dir
        self.embeddings_dir = workspace_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.embeddings_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="schema_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_schema(self, schema_info, use_semantic: bool = True) -> None:
        """
        Index schema information into vector store.
        
        Args:
            schema_info: SchemaInfo object
            use_semantic: If True, use semantic documents; else use basic text chunks
        """
        # Check if semantic profile exists
        semantic_profile_path = self.workspace_dir / "semantic_profile.json"
        
        if use_semantic and semantic_profile_path.exists():
            print("📚 Using semantic documents for indexing...")
            self._index_semantic_documents(semantic_profile_path)
        else:
            print("📚 Using basic text chunks for indexing...")
            chunks = schema_info.to_text_chunks()
            self._index_basic_chunks(chunks)
    
    def _index_semantic_documents(self, semantic_profile_path: Path) -> None:
        """Index rich semantic documents."""
        # Load semantic profile
        with open(semantic_profile_path, 'r', encoding='utf-8') as f:
            semantic_data = json.load(f)
        
        # Generate semantic documents
        from sqlgenx.core.semantic_enricher import SemanticSchema, SemanticDocumentGenerator
        
        # Reconstruct semantic schema (simplified)
        documents = []
        
        # Index table semantics
        for table_sem in semantic_data.get('table_semantics', []):
            doc_content = self._build_rich_document(table_sem)
            
            documents.append({
                'id': f"table_{table_sem['table_name']}",
                'content': doc_content,
                'metadata': {
                    'table': table_sem['table_name'],
                    'type': 'table_semantic',
                    'domain': table_sem.get('business_domain', 'unknown'),
                    'table_type': table_sem.get('table_type', 'unknown')
                }
            })
        
        # Index domain groupings
        for domain, tables in semantic_data.get('domain_groupings', {}).items():
            doc_content = f"Domain: {domain}\nTables: {', '.join(tables)}"
            
            documents.append({
                'id': f"domain_{domain}",
                'content': doc_content,
                'metadata': {
                    'table': f"_domain_{domain}",
                    'type': 'domain',
                    'domain': domain
                }
            })
        
        # Index join paths
        for i, join in enumerate(semantic_data.get('join_paths', [])):
            doc_content = (
                f"Join: {join['from_table']} to {join['to_table']}\n"
                f"Type: {join['join_type']}\n"
                f"Description: {join.get('description', '')}"
            )
            
            documents.append({
                'id': f"join_{i}",
                'content': doc_content,
                'metadata': {
                    'table': f"{join['from_table']},{join['to_table']}",
                    'type': 'join_path',
                    'from_table': join['from_table'],
                    'to_table': join['to_table']
                }
            })
        
        # Clear and index
        self._clear_and_index(documents)
    
    def _build_rich_document(self, table_sem: Dict[str, Any]) -> str:
        """Build rich document text from table semantics."""
        lines = []
        
        table_name = table_sem['table_name']
        lines.append(f"Table: {table_name}")
        lines.append(f"Type: {table_sem.get('table_type', 'unknown')}")
        lines.append(f"Domain: {table_sem.get('business_domain', 'general')}")
        lines.append(f"Purpose: {table_sem.get('business_purpose', '')}")
        lines.append("")
        
        # Use cases
        use_cases = table_sem.get('use_cases', [])
        if use_cases:
            lines.append("Use Cases:")
            for uc in use_cases:
                lines.append(f"- {uc}")
            lines.append("")
        
        # Fields
        lines.append("Fields:")
        for field in table_sem.get('field_semantics', []):
            field_line = f"- {field['column_name']}: {field.get('business_meaning', field['column_name'])}"
            
            synonyms = field.get('synonyms', [])
            if synonyms:
                field_line += f" (synonyms: {', '.join(synonyms)})"
            
            field_line += f" [{field.get('data_category', 'unknown')}]"
            lines.append(field_line)
        
        lines.append("")
        
        # Derived fields / metrics
        derived = table_sem.get('derived_fields', [])
        if derived:
            lines.append("Available Metrics:")
            for df in derived:
                lines.append(f"- {df['name']}: {df.get('description', '')}")
        
        lines.append("")
        
        # Related tables
        related = table_sem.get('related_tables', [])
        if related:
            lines.append(f"Related to: {', '.join(related)}")
        
        # Keywords
        keywords = []
        keywords.append(table_name)
        keywords.append(table_name.replace('_', ' '))
        keywords.extend(f['column_name'] for f in table_sem.get('field_semantics', []))
        keywords.extend(table_sem.get('common_metrics', []))
        
        lines.append("")
        lines.append(f"Keywords: {', '.join(set(keywords))}")
        
        return "\n".join(lines)
    
    def _index_basic_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """Index basic text chunks (fallback method)."""
        if not chunks:
            return
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f"chunk_{i}",
                'content': chunk["content"],
                'metadata': {
                    'table': chunk["table"],
                    'type': chunk["type"]
                }
            })
        
        self._clear_and_index(documents)
    
    def _clear_and_index(self, documents: List[Dict[str, Any]]) -> None:
        """Clear existing embeddings and index new documents."""
        # Clear existing embeddings
        try:
            self.client.delete_collection("schema_embeddings")
            self.collection = self.client.create_collection(
                name="schema_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass
        
        # Prepare data for ChromaDB
        ids = []
        doc_texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc['id'])
            doc_texts.append(doc['content'])
            metadatas.append(doc['metadata'])
        
        # Add to collection
        if doc_texts:
            self.collection.add(
                ids=ids,
                documents=doc_texts,
                metadatas=metadatas
            )
            
            print(f"   ✓ Indexed {len(doc_texts)} documents")
    
    def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant schema context for a query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        except Exception as e:
            print(f"Warning: Failed to retrieve context: {e}")
            return []
        
        contexts = []
        
        if results and results.get("documents") and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = {}
                distance = 0.0
                
                if results.get("metadatas") and results["metadatas"]:
                    metadata = results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
                
                if results.get("distances") and results["distances"]:
                    distance = results["distances"][0][i] if i < len(results["distances"][0]) else 0.0
                
                contexts.append({
                    "content": doc,
                    "table": metadata.get("table", "unknown"),
                    "type": metadata.get("type", "unknown"),
                    "relevance_score": 1 - distance  # Convert distance to similarity
                })
        
        return contexts
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the schema."""
        try:
            results = self.collection.get()
        except Exception:
            return []
        
        if not results or not results.get("metadatas"):
            return []
        
        tables = set()
        for metadata in results["metadatas"]:
            table = metadata.get("table", "")
            # Skip special documents
            if table and not table.startswith('_'):
                tables.add(table)
        
        return sorted(list(tables))
    
    def get_table_context(self, table_name: str) -> str:
        """Get full context for a specific table."""
        try:
            results = self.collection.get(
                where={"table": table_name}
            )
            
            if results and results.get("documents"):
                return "\n".join(results["documents"])
        except Exception:
            pass
        
        return ""