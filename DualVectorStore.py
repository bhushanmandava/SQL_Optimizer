import logging
from typing import Dict, List
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, Embeddings, Metadatas, IDs

from Schema import logger, ASTShape, OptimizationResult, SQLQuery


class ChromaVectorStore:
    """Chroma vector storage for similarity search and metadata filtering"""

    def __init__(self, chroma_config: Dict):
        try:
            self.client = chromadb.Client()
            self.collection_name = chroma_config.get("collection", "sql_queries")

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def store_query(self, query: SQLQuery, optimization_result: OptimizationResult):
        try:
            metadata = {
                "original_sql": query.original_sql,
                "optimized_sql": optimization_result.optimized_sql,
                "ast_shape": query.ast_shape.value,
                "table_count": query.ast_features.get('table_count', 0),
                "join_count": query.ast_features.get('join_count', 0),
                "complexity_score": query.ast_features.get('complexity_score', 0.0),
                "optimization_type": optimization_result.optimization_type.value,
                "confidence_score": optimization_result.confidence_score,
                "timestamp": datetime.now().isoformat()
            }

            self.collection.add(
                documents=[query.original_sql],
                embeddings=[query.embedding],
                metadatas=[metadata],
                ids=[query.query_hash]
            )
            logger.info(f"Stored query {query.query_hash} in ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to store query in ChromaDB: {e}")

    def search_similar_queries(self, query_embedding: List[float], ast_shape: ASTShape, limit: int = 50) -> List[Dict]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"ast_shape": ast_shape.value}
            )
            return [
                {
                    "id": result_id,
                    "score": score,
                    "metadata": metadata
                }
                for result_id, score, metadata in zip(
                    results["ids"][0],
                    results["distances"][0],
                    results["metadatas"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Failed to search similar queries in ChromaDB: {e}")
            return []

    def search_by_metadata(self, filters: Dict, limit: int = 10) -> List[Dict]:
        try:
            chroma_filter = self._build_chroma_filter(filters)

            results = self.collection.get(where=chroma_filter)
            matches = zip(results["ids"], results["metadatas"])
            return [
                {"id": query_id, "metadata": metadata}
                for query_id, metadata in matches
            ][:limit]
        except Exception as e:
            logger.error(f"Failed to search by metadata in ChromaDB: {e}")
            return []

    def _build_chroma_filter(self, filters: Dict) -> Dict:
        chroma_filter = {}
        if "ast_shape" in filters:
            chroma_filter["ast_shape"] = filters["ast_shape"]
        if "min_confidence" in filters:
            chroma_filter["confidence_score"] = {"$gte": filters["min_confidence"]}
        if "complexity_range" in filters:
            chroma_filter["complexity_score"] = {
                "$gte": filters["complexity_range"][0],
                "$lte": filters["complexity_range"][1]
            }
        return chroma_filter
