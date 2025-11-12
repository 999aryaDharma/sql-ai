"""Vector store integration using ChromaDB."""
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings


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
    
    def index_schema(self, schema_info) -> None:
        """Index schema information into vector store."""
        chunks = schema_info.to_text_chunks()
        
        if not chunks:
            return
        
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
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk["content"])
            metadatas.append({
                "table": chunk["table"],
                "type": chunk["type"]
            })
        
        # Add to collection
        if documents:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
    
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
            if metadata.get("table"):
                tables.add(metadata["table"])
        
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