#!/usr/bin/env python3
"""
Main execution script for the LLM-R2 SQL Optimization System
Demonstrates the complete workflow of SQL query optimization using hybrid embeddings,
rule-based transformations, and dual vector storage.
"""

import logging
import yaml
import hashlib
from datetime import datetime
from typing import Dict, List

# Import all the system components
from LLMR2RuleEngine import LLMR2RuleEngine
from EnhancedSqlGlotAnalyzer import EnhancedSQLGlotAnalyzer
from HybridEmbeddingGenerator import HybridEmbeddingGenerator
from DualVectorStore import ChromaVectorStore
from Schema import SQLQuery, OptimizationResult, OptimizationType, ASTShape, logger

# Assuming these are defined in Schema.py
# try:
    
# except ImportError:
#     # Fallback definitions if Schema.py is not available
#     from enum import Enum
#     from dataclasses import dataclass
#     from typing import Optional
    
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     class ASTShape(Enum):
#         SIMPLE_SELECT = "simple_select"
#         JOIN_HEAVY = "join_heavy"
#         SUBQUERY_COMPLEX = "subquery_complex"
#         AGGREGATE_WINDOW = "aggregate_window"
#         CTE_RECURSIVE = "cte_recursive"
#         UNION_MULTI = "union_multi"
    
#     class OptimizationType(Enum):
#         LLM_R2_REWRITE = "llm_r2_rewrite"
#         TRADITIONAL = "traditional"
#         HYBRID = "hybrid"
    
#     @dataclass
#     class SQLQuery:
#         original_sql: str
#         query_hash: str
#         embedding: List[float]
#         ast_features: Dict
#         ast_shape: ASTShape
#         timestamp: str
    
#     @dataclass
#     class OptimizationResult:
#         optimized_sql: str
#         optimization_type: OptimizationType
#         confidence_score: float
#         explanation: str
#         optimization_stages: List[str]
#         llm_reasoning: str


class SQLOptimizationSystem:
    """Main orchestrator for the SQL optimization pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for the system"""
        return {
            'embedding': {
                'codebert_weight': 0.6,
                # 't5_weight': 0.3,
                'graph_weight': 0.4
            },
            'qdrant': {
                'host': 'localhost',
                'port': 6333,
                'collection': 'sql_queries'
            },
            'weaviate': {
                'url': 'http://localhost:8081',
                'class': 'SqlQuery'
            },
            'optimization': {
                'similarity_threshold': 0.7,
                'confidence_threshold': 0.6,
                'max_similar_queries': 10
            }
        }
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Core analyzers
            self.sql_analyzer = EnhancedSQLGlotAnalyzer()
            self.rule_engine = LLMR2RuleEngine()
            # **** we also need to add a RAG system in future other that the Current rule based optimization#
            # Embedding system
            self.embedding_generator = HybridEmbeddingGenerator(
                self.config.get('embedding', {})
            )
            
            # Vector storage (optional - requires external services)
            #  ************* the vector store is goin to be turned into Chroma DB
            try:
                self.vector_store = ChromaVectorStore( 
                    self.config.get('chroma', {}),
                    # self.config.get('weaviate', {})
                )
                self.vector_store_available = True
                logger.info("Vector store initialized successfully")
            except Exception as e:
                logger.warning(f"Vector store not available: {e}")
                self.vector_store_available = False
            
            logger.info("SQL Optimization System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def optimize_query(self, sql: str, dialect: str = 'mysql') -> Dict:
        """Main optimization pipeline for a single SQL query"""
        try:
            logger.info(f"Starting optimization for query: {sql[:100]}...")
            
            # Step 1: Normalize and analyze SQL
            normalized_sql = self.sql_analyzer.normalize_sql(sql, dialect)
            ast_features = self.sql_analyzer.extract_ast_features(normalized_sql, dialect)
            ast_shape = self.sql_analyzer.classify_ast_shape(ast_features)
            
            logger.info(f"Query analysis complete - Shape: {ast_shape.value}, "
                       f"Complexity: {ast_features.get('complexity_score', 0):.2f}") # step to calculate the complety of the quuey we are goin to optimize
            
            # Step 2: Generate embedding
            embedding = self.embedding_generator.generate_embedding(normalized_sql, ast_features)
            
            # Step 3: Create SQLQuery object
            query_hash = hashlib.md5(normalized_sql.encode()).hexdigest()
            sql_query = SQLQuery(
                normalized_sql=normalized_sql,
                query_hash=query_hash,
                embedding=embedding,
                ast_features=ast_features,
                ast_shape=ast_shape,
                # timestamp=datetime.now().isoformat()
            )
            
            # Step 4: Find similar queries (if vector store available)
            similar_optimizations = []
            if self.vector_store_available:
                similar_queries = self.vector_store.search_similar_queries(
                    embedding, ast_shape, 
                    limit=self.config.get('optimization', {}).get('max_similar_queries', 10)
                )
                similar_optimizations = [
                    {
                        'original_sql': q['payload']['original_sql'],
                        'optimized_sql': q['payload']['optimized_sql'],
                        'similarity_score': q['score'],
                        'optimization_type': q['payload']['optimization_type']
                    }
                    for q in similar_queries
                    if q['score'] >= self.config.get('optimization', {}).get('similarity_threshold', 0.7)
                ]
                logger.info(f"Found {len(similar_optimizations)} similar optimized queries")
            
            # Step 5: Apply rule-based optimization
            optimization_result = self.rule_engine.apply_rules(
                normalized_sql, ast_features, similar_optimizations
            )
            
            if optimization_result:
                logger.info(f"Optimization successful - Confidence: {optimization_result.confidence_score:.2f}")
                
                # Step 6: Store the optimization (if vector store available)
                if self.vector_store_available:
                    try:
                        self.vector_store.store_query(sql_query, optimization_result)
                        logger.info("Optimization stored in vector database")
                    except Exception as e:
                        logger.warning(f"Failed to store optimization: {e}")
                
                return {
                    'success': True,
                    'original_sql': sql,
                    'normalized_sql': normalized_sql,
                    'optimized_sql': optimization_result.optimized_sql,
                    'optimization_type': optimization_result.optimization_type.value,
                    'confidence_score': optimization_result.confidence_score,
                    'explanation': optimization_result.explanation,
                    'optimization_stages': optimization_result.optimization_stages,
                    'llm_reasoning': optimization_result.llm_reasoning,
                    'ast_features': ast_features,
                    'ast_shape': ast_shape.value,
                    'similar_queries_found': len(similar_optimizations)
                }
            else:
                logger.info("No optimization rules applied")
                return {
                    'success': False,
                    'original_sql': sql,
                    'normalized_sql': normalized_sql,
                    'message': 'No applicable optimization rules found',
                    'ast_features': ast_features,
                    'ast_shape': ast_shape.value
                }
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'success': False,
                'original_sql': sql,
                'error': str(e)
            }
    # ************ for now it is out of scope
    def batch_optimize(self, queries: List[str], dialect: str = 'mysql') -> List[Dict]:
        """Optimize multiple queries in batch"""
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.optimize_query(query, dialect)
            results.append(result)
        return results
    
    def search_similar_optimizations(self, sql: str, limit: int = 5) -> List[Dict]:
        """Search for similar previously optimized queries"""
        if not self.vector_store_available:
            return []
        
        try:
            # Analyze the input query
            normalized_sql = self.sql_analyzer.normalize_sql(sql)
            ast_features = self.sql_analyzer.extract_ast_features(normalized_sql)
            ast_shape = self.sql_analyzer.classify_ast_shape(ast_features)
            embedding = self.embedding_generator.generate_embedding(normalized_sql, ast_features)
            
            # Search for similar queries
            similar_queries = self.vector_store.search_similar_queries(
                embedding, ast_shape, limit=limit
            )
            
            return [
                {
                    'original_sql': q['payload']['original_sql'],
                    'optimized_sql': q['payload']['optimized_sql'],
                    'similarity_score': q['score'],
                    'optimization_type': q['payload']['optimization_type'],
                    'confidence_score': q['payload']['confidence_score']
                }
                for q in similar_queries
            ]
            
        except Exception as e:
            logger.error(f"Failed to search similar optimizations: {e}")
            return []


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        'embedding': {
            'codebert_weight': 0.4,
            # 't5_weight': 0.3,
            'graph_weight': 0.3
        },
        'qdrant': {
            'host': 'localhost',
            'port': 6333,
            'collection': 'sql_queries'
        },
        'weaviate': {
            'url': 'http://localhost:8080',
            'class': 'SqlQuery'
        },
        'optimization': {
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.6,
            'max_similar_queries': 10
        }
    }
    ################future change going to be working with Chroma Db
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("Created sample config.yaml file")


def main():
    """Main execution function with example usage"""
    print("🚀 Starting LLM-R2 SQL Optimization System")
    
    # Create sample config if it doesn't exist
    try:
        with open('config.yaml', 'r'):
            pass
    except FileNotFoundError:
        create_sample_config()
    
    # Initialize the system
    try:
        optimizer = SQLOptimizationSystem()
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return
    
    # Example queries for testing
    test_queries = [
        # Predicate pushdown
        "SELECT * FROM (SELECT * FROM orders) AS sub WHERE sub.order_date > '2023-01-01'",

        # Join reordering
        "SELECT * FROM large_table AS lt JOIN small_table AS st ON lt.id = st.id JOIN medium_table AS mt ON mt.id = st.mid",

        # EXISTS to JOIN (subquery)
        "SELECT customer_id FROM customers AS c WHERE EXISTS (SELECT 1 FROM orders AS o WHERE o.customer_id = c.customer_id)",

        # Redundant DISTINCT
        "SELECT DISTINCT customer_id FROM orders GROUP BY customer_id",

        # Multiple rules: DISTINCT + EXISTS + JOIN + GROUP BY
        """
        SELECT DISTINCT c.customer_name, 
            AVG(o.total_amount) AS avg_order
        FROM customers AS c 
        JOIN orders AS o ON c.customer_id = o.customer_id
        WHERE EXISTS (
            SELECT 1 FROM order_items AS oi 
            WHERE oi.order_id = o.order_id 
            AND oi.product_id IN (SELECT product_id FROM products WHERE category = 'Electronics')
        )
        GROUP BY c.customer_name, c.customer_id
        ORDER BY avg_order DESC
        """
    ]

    
    print(f"\n📊 Testing with {len(test_queries)} sample queries...")
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Processing Query #{i}")
        print(f"{'='*60}")
        print(f"Original SQL:\n{query}\n")
        
        result = optimizer.optimize_query(query)
        
        if result['success']:
            print("✅ Optimization successful!")
            print(f"🎯 AST Shape: {result['ast_shape']}")
            print(f"📈 Confidence Score: {result['confidence_score']:.2f}")
            print(f"🔧 Applied Rules: {', '.join(result['optimization_stages'])}")
            print(f"📝 Explanation: {result['explanation']}")
            print(f"\nOptimized SQL:\n{result['optimized_sql']}")
            
            if result['similar_queries_found'] > 0:
                print(f"🔄 Found {result['similar_queries_found']} similar optimized queries")
        else:
            print("❌ No optimization applied")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Reason: {result.get('message', 'Unknown')}")
    
    # Demonstrate batch processing
    print(f"\n{'='*60}")
    print("🔄 Demonstrating batch processing...")
    print(f"{'='*60}")
    
    batch_results = optimizer.batch_optimize(test_queries[:3])
    successful_optimizations = sum(1 for r in batch_results if r['success'])
    print(f"✅ Successfully optimized {successful_optimizations}/{len(batch_results)} queries")
    
    # Demonstrate similarity search
    if optimizer.vector_store_available:
        print(f"\n{'='*60}")
        print("🔍 Demonstrating similarity search...")
        print(f"{'='*60}")
        
        search_query = "SELECT customer_id FROM customers WHERE customer_id IN (SELECT customer_id FROM orders)"
        similar = optimizer.search_similar_optimizations(search_query, limit=3)
        
        if similar:
            print(f"Found {len(similar)} similar optimizations:")
            for sim in similar:
                print(f"  - Similarity: {sim['similarity_score']:.3f}, "
                      f"Type: {sim['optimization_type']}")
        else:
            print("No similar optimizations found (vector store may be empty)")
    
    print(f"\n{'='*60}")
    print("🎉 SQL Optimization System demonstration complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()