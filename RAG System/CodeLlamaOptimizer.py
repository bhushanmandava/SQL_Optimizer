import time
import torch
import logging
import json
import hashlib
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModel
from enum import Enum

# --- ENUMS & DATA CLASSES ---

class OptimizationType(Enum):
    LLM_R2_REWRITE = "LLM_R2_REWRITE"
    CODELLAMA_REFINEMENT = "CODELLAMA_REFINEMENT"

@dataclass
class OptimizationResult:
    optimized_sql: str
    optimization_type: OptimizationType
    confidence_score: float
    explanation: str
    performance_gain_estimate: float
    optimization_stages: List[str]
    llm_reasoning: str

@dataclass
class SQLQuery:
    original_sql: str
    query_hash: str
    ast_features: Dict
    ast_shape: str
    embedding: Any
    normalized_sql: str
    complexity_score: float
    table_references: List[str]
    join_count: int

# --- LOGGER SETUP ---

logger = logging.getLogger("CodeLlamaOptimizer")
logging.basicConfig(level=logging.INFO)

# --- METRICS STUBS ---

class DummyMetric:
    def inc(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass
    def observe(self, *args, **kwargs): pass

optimization_requests = DummyMetric()
optimization_latency = DummyMetric()
llm_inference_latency = DummyMetric()
cache_hits = DummyMetric()
accuracy_gauge = DummyMetric()

# --- CODELLAMA OPTIMIZER ---

class CodeLlamaOptimizer:
    """Fine-tuned CodeLlama optimizer for advanced SQL refinement"""

    def __init__(self, model_config: Dict):
        self.model_name = model_config.get('model_name', 'codellama/CodeLlama-34b-Instruct-hf')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        # Fine-tuning parameters
        self.max_length = model_config.get('max_length', 2048)
        self.temperature = model_config.get('temperature', 0.1)
        self.top_p = model_config.get('top_p', 0.9)

    def optimize_sql(self, sql: str, context: Dict, previous_optimizations: List[OptimizationResult]) -> Optional[OptimizationResult]:
        """Apply CodeLlama-based optimization with context awareness"""
        start_time = time.time()
        try:
            # Prepare optimization prompt
            prompt = self._build_optimization_prompt(sql, context, previous_optimizations)
            # Generate optimized SQL
            optimized_sql, reasoning = self._generate_optimization(prompt)
            # Calculate confidence based on model certainty and context alignment
            confidence = self._calculate_confidence(sql, optimized_sql, context)
            # Estimate performance gain
            performance_gain = self._estimate_performance_gain(sql, optimized_sql, context)
            return OptimizationResult(
                optimized_sql=optimized_sql,
                optimization_type=OptimizationType.CODELLAMA_REFINEMENT,
                confidence_score=confidence,
                explanation="CodeLlama fine-tuned optimization with contextual analysis",
                performance_gain_estimate=performance_gain,
                optimization_stages=["codellama_analysis", "contextual_refinement"],
                llm_reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"CodeLlama optimization failed: {e}")
            return None
        finally:
            llm_inference_latency.observe(time.time() - start_time)

    def _build_optimization_prompt(self, sql: str, context: Dict, previous_optimizations: List[OptimizationResult]) -> str:
        """Build comprehensive optimization prompt for CodeLlama"""
        # Extract context information
        ast_features = context.get('ast_features', {})
        similar_patterns = context.get('similar_patterns', [])
        prompt = f"""[INST] You are an expert SQL optimization system.
TASK: Optimize the following SQL query for performance while maintaining correctness.
ORIGINAL QUERY:
{sql}

QUERY ANALYSIS:
- Tables: {ast_features.get('table_count', 0)}
- Joins: {ast_features.get('join_count', 0)}
- Subqueries: {ast_features.get('subquery_count', 0)}
- Complexity Score: {ast_features.get('complexity_score', 0):.1f}
- AST Shape: {context.get('ast_shape', 'unknown')}
PREVIOUS OPTIMIZATIONS APPLIED:
"""
        for opt in previous_optimizations:
            prompt += f"- {opt.optimization_type.value}: {opt.explanation}\n"
        if similar_patterns:
            prompt += f"\nSIMILAR OPTIMIZATION PATTERNS:\n"
            for pattern in similar_patterns[:3]:  # Limit to top 3
                prompt += f"- Pattern: {pattern.get('optimization_type', 'unknown')}\n"
                prompt += f" Confidence: {pattern.get('confidence_score', 0):.2f}\n"
        prompt += """
OPTIMIZATION GUIDELINES:
1. Focus on join order optimization and predicate pushdown
2. Consider index-friendly query structures
3. Eliminate redundant operations
4. Optimize subqueries and CTEs
5. Ensure logical equivalence
Provide the optimized SQL and explain your reasoning.
OPTIMIZED QUERY:"""
        return prompt

    def _generate_optimization(self, prompt: str) -> Tuple[str, str]:
        """Generate optimized SQL using CodeLlama"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract optimized SQL and reasoning
            optimized_sql, reasoning = self._parse_model_response(response)
            return optimized_sql, reasoning
        except Exception as e:
            logger.error(f"CodeLlama generation failed: {e}")
            return "", f"Generation failed: {str(e)}"

    def _parse_model_response(self, response: str) -> Tuple[str, str]:
        """Parse CodeLlama response to extract SQL and reasoning"""
        try:
            # Look for SQL code blocks
            import re
            sql_pattern = r'``````'
            sql_matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
            if sql_matches:
                optimized_sql = sql_matches[-1].strip()  # Take the last SQL block
            else:
                # Fallback: extract everything after "OPTIMIZED QUERY:"
                parts = response.split("OPTIMIZED QUERY:")
                if len(parts) > 1:
                    optimized_sql = parts[-1].strip()
                else:
                    optimized_sql = ""
            # Extract reasoning (everything before the SQL)
            reasoning_parts = response.split("```
            reasoning = reasoning_parts.strip() if reasoning_parts else "No reasoning provided"
            return optimized_sql, reasoning
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return "", f"Parsing failed: {str(e)}"

    def _calculate_confidence(self, original_sql: str, optimized_sql: str, context: Dict) -> float:
        """Calculate confidence score for CodeLlama optimization"""
        if not optimized_sql or optimized_sql == original_sql:
            return 0.1
        # Base confidence from successful generation
        confidence = 0.6
        # Boost for structural improvements
        try:
            # Stub for parse_one function - replace with actual implementation
            def parse_one(sql):
                return sql
                
            original_ast = parse_one(original_sql)
            optimized_ast = parse_one(optimized_sql)
            if original_ast and optimized_ast:
                # Compare complexity
                original_complexity = len(str(original_ast))
                optimized_complexity = len(str(optimized_ast))
                if optimized_complexity < original_complexity:
                    confidence += 0.2
                # Check for common optimization patterns
                if "JOIN" in optimized_sql and context.get('join_count', 0) > 1:
                    confidence += 0.1
                if "WHERE" in optimized_sql and "subquery" in original_sql.lower():
                    confidence += 0.1
        except:
            pass
        return min(0.95, confidence)

    def _estimate_performance_gain(self, original_sql: str, optimized_sql: str, context: Dict) -> float:
        """Estimate performance improvement percentage"""
        try:
            # Simple heuristics for performance estimation
            gain = 0.0
            # Join optimizations
            if context.get('join_count', 0) > 2:
                if "INNER JOIN" in optimized_sql and "LEFT JOIN" in original_sql:
                    gain += 15.0
            # Subquery optimizations
            original_subqueries = original_sql.lower().count('select')
            optimized_subqueries = optimized_sql.lower().count('select')
            if optimized_subqueries < original_subqueries:
                gain += 20.0
            # WHERE clause improvements
            if optimized_sql.count('WHERE') > original_sql.count('WHERE'):
                gain += 10.0
            # Index-friendly patterns
            if "ORDER BY" in optimized_sql and "LIMIT" in optimized_sql:
                gain += 8.0
            return min(50.0, gain)
        except Exception as e:
            logger.error(f"Performance estimation failed: {e}")
            return 5.0  # Conservative default

# --- CACHE ---

class OptimizationCache:
    def __init__(self, redis_config: Dict):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
        except ImportError:
            # Fallback to in-memory cache if redis not available
            self.redis_client = None
            self.store = {}
            
        self.ttl = redis_config.get('ttl', 3600)  # 1 hour default

    def get(self, query_hash: str) -> Optional[OptimizationResult]:
        """Get cached optimization result"""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(f"opt:{query_hash}")
            else:
                cached_data = self.store.get(query_hash)
                
            if cached_data:
                cache_hits.inc()
                if isinstance(cached_data, str):
                    data = json.loads(cached_data)
                else:
                    data = cached_data
                # Reconstruct OptimizationResult object
                data['optimization_type'] = OptimizationType(data['optimization_type'])
                return OptimizationResult(**data)
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
        return None

    def set(self, query_hash: str, result: OptimizationResult):
        """Cache optimization result"""
        try:
            # Convert to dict for JSON serialization
            data = asdict(result)
            data['optimization_type'] = result.optimization_type.value
            
            if self.redis_client:
                self.redis_client.setex(
                    f"opt:{query_hash}",
                    self.ttl,
                    json.dumps(data)
                )
            else:
                self.store[query_hash] = data
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")

# --- VECTOR STORE STUBS ---

class DualVectorStore:
    def __init__(self, qdrant_config: Dict, weaviate_config: Dict):
        # Initialize vector stores
        pass
        
    def search_similar_queries(self, embedding, ast_shape, limit=50):
        # Stub implementation
        return []
        
    def search_by_metadata(self, metadata_filters):
        # Stub implementation
        return []
        
    def store_query(self, query: SQLQuery, result: OptimizationResult):
        # Stub implementation
        pass

# --- EMBEDDING GENERATOR STUB ---

class HybridEmbeddingGenerator:
    def __init__(self, config: Dict):
        # Initialize embedding models
        pass
        
    def generate_embedding(self, normalized_sql, ast_features):
        # Stub implementation
        return [0.0] * 768

# --- ANALYZER STUB ---

class EnhancedSQLGlotAnalyzer:
    def normalize_sql(self, sql, dialect):
        # Stub implementation
        return sql
        
    def extract_ast_features(self, sql, dialect):
        # Stub implementation
        return {
            "table_count": 1, 
            "join_count": 0, 
            "subquery_count": 0, 
            "complexity_score": 1.0, 
            "table_references": []
        }
        
    def classify_ast_shape(self, ast_features):
        # Stub implementation
        return "simple"
        
    def compute_ast_similarity(self, sql1, sql2):
        # Stub implementation
        return 0.8

# --- RULE ENGINE STUB ---

class LLMR2RuleEngine:
    def __init__(self, rules_config):
        # Initialize rule engine
        pass
        
    def apply_rules(self, sql, ast_features, similar_queries):
        # Stub implementation
        return OptimizationResult(
            optimized_sql=sql,
            optimization_type=OptimizationType.LLM_R2_REWRITE,
            confidence_score=0.7,
            explanation="Rule-based rewrite",
            performance_gain_estimate=10.0,
            optimization_stages=["rule_engine"],
            llm_reasoning="Applied rule-based optimization"
        )

# --- MAIN OPTIMIZER SYSTEM ---

class ComprehensiveSQLOptimizer:
    def __init__(self, config: Dict):
        # Initialize components
        self.analyzer = EnhancedSQLGlotAnalyzer()
        self.embedding_generator = HybridEmbeddingGenerator(config.get('embedding', {}))
        self.vector_store = DualVectorStore(
            config.get('qdrant', {}),
            config.get('weaviate', {})
        )
        self.cache = OptimizationCache(config.get('redis', {}))
        self.llm_r2_engine = LLMR2RuleEngine(config.get('rules_config', 'sql_rules.yaml'))
        self.codellama_optimizer = CodeLlamaOptimizer(config.get('codellama', {}))
        
        # Configuration
        self.use_cache = config.get('use_cache', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.max_similar_queries = config.get('max_similar_queries', 50)
        
        # Start metrics server if configured
        if config.get('metrics_port'):
            try:
                from prometheus_client import start_http_server
                start_http_server(config['metrics_port'])
            except ImportError:
                logger.warning("prometheus_client not installed, metrics server not started")
                
        logger.info("Comprehensive SQL Optimization System initialized")

    async def optimize_sql(self, sql: str, dialect: str = 'mysql',
                          optimization_types: List[OptimizationType] = None) -> List[OptimizationResult]:
        """Main optimization pipeline following the architecture diagram"""
        optimization_requests.inc()
        start_time = time.time()
        try:
            # Step 1: Normalization with SQLGlot
            logger.info("Step 1: Normalizing SQL with SQLGlot")
            normalized_sql = self.analyzer.normalize_sql(sql, dialect)
            
            # Generate query hash for caching
            query_hash = hashlib.md5(normalized_sql.encode()).hexdigest()
            
            # Check cache first
            if self.use_cache:
                cached_result = self.cache.get(query_hash)
                if cached_result:
                    logger.info("Returning cached optimization result")
                    return [cached_result]
            
            # Step 2: Vector Embedding Lookup
            logger.info("Step 2: Generating hybrid embeddings")
            ast_features = self.analyzer.extract_ast_features(sql, dialect)
            ast_shape = self.analyzer.classify_ast_shape(ast_features)
            embedding = self.embedding_generator.generate_embedding(normalized_sql, ast_features)
            
            # Create SQLQuery object
            query = SQLQuery(
                original_sql=sql,
                query_hash=query_hash,
                ast_features=ast_features,
                ast_shape=ast_shape,
                embedding=embedding,
                normalized_sql=normalized_sql,
                complexity_score=ast_features.get('complexity_score', 0.0),
                table_references=ast_features.get('table_references', []),
                join_count=ast_features.get('join_count', 0)
            )
            
            # Step 3: Dual Vector Search (Qdrant + Weaviate)
            logger.info("Step 3: Searching for similar queries in dual vector stores")
            # Qdrant: Find Similar Queries
            qdrant_results = self.vector_store.search_similar_queries(
                embedding, ast_shape, limit=self.max_similar_queries
            )
            
            # Weaviate: Metadata Filtered Search
            metadata_filters = {
                "ast_shape": ast_shape,
                "complexity_range": [
                    max(0, ast_features.get('complexity_score', 0) - 10),
                    ast_features.get('complexity_score', 0) + 10
                ],
                "min_confidence": 0.7
            }
            weaviate_results = self.vector_store.search_by_metadata(metadata_filters)
            
            # Step 4: AST Shape Filtering
            logger.info("Step 4: Applying AST shape filtering")
            filtered_similar_queries = self._apply_ast_filtering(
                qdrant_results, weaviate_results, query
            )
            
            # Determine optimization types if not specified
            if optimization_types is None:
                optimization_types = self._determine_optimization_strategy(query, filtered_similar_queries)
                
            results = []
            
            # Step 5: Rule-based Rewrite with LLM-R2
            if OptimizationType.LLM_R2_REWRITE in optimization_types:
                logger.info("Step 5: Applying LLM-R2 rule-based optimization")
                llm_r2_result = self.llm_r2_engine.apply_rules(
                    sql, ast_features, filtered_similar_queries
                )
                if llm_r2_result and llm_r2_result.confidence_score >= self.confidence_threshold:
                    results.append(llm_r2_result)
            
            # Step 6: Fine-Tuned CodeLlama Rewrite
            if OptimizationType.CODELLAMA_REFINEMENT in optimization_types:
                logger.info("Step 6: Applying CodeLlama fine-tuned optimization")
                # Build context for CodeLlama
                context = {
                    'ast_features': ast_features,
                    'ast_shape': ast_shape,
                    'similar_patterns': filtered_similar_queries,
                    'query_hash': query_hash
                }
                codellama_result = self.codellama_optimizer.optimize_sql(
                    sql, context, results
                )
                if codellama_result and codellama_result.confidence_score >= self.confidence_threshold:
                    results.append(codellama_result)
            
            # Step 7: Return Optimized SQL
            logger.info("Step 7: Finalizing and returning optimization results")
            
            # Store successful optimizations for future use
            for result in results:
                if result.confidence_score >= self.confidence_threshold:
                    self.vector_store.store_query(query, result)
                    if self.use_cache:
                        self.cache.set(query_hash, result)
            
            # Update metrics
            if results:
                max_confidence = max(r.confidence_score for r in results)
                accuracy_gauge.set(max_confidence)
                
            return results
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}")
            return []
        finally:
            optimization_latency.observe(time.time() - start_time)

    def _apply_ast_filtering(self, qdrant_results: List[Dict],
                           weaviate_results: List[Dict], query: SQLQuery) -> List[Dict]:
        """Apply AST shape filtering to combine results from both vector stores"""
        combined_results = []
        seen_queries = set()
        
        # Process Qdrant results (vector similarity)
        for result in qdrant_results:
            query_id = result.get('id')
            if query_id not in seen_queries:
                # Calculate AST similarity
                original_sql = result.get('payload', {}).get('original_sql', '')
                ast_similarity = self.analyzer.compute_ast_similarity(
                    query.original_sql, original_sql
                )
                if ast_similarity >= 0.3:  # Threshold for AST similarity
                    result['ast_similarity'] = ast_similarity
                    result['source'] = 'qdrant'
                    combined_results.append(result)
                    seen_queries.add(query_id)
        
        # Process Weaviate results (metadata filtering)
        for result in weaviate_results:
            query_hash = result.get('queryHash')
            if query_hash and query_hash not in seen_queries:
                # Convert Weaviate format to standard format
                formatted_result = {
                    'id': query_hash,
                    'score': result.get('confidenceScore', 0.0),
                    'payload': {
                        'original_sql': result.get('originalSql', ''),
                        'optimized_sql': result.get('optimizedSql', ''),
                        'optimization_type': result.get('optimizationType', ''),
                        'confidence_score': result.get('confidenceScore', 0.0)
                    },
                    'ast_similarity': 0.8,  # High similarity due to metadata matching
                    'source': 'weaviate'
                }
                combined_results.append(formatted_result)
                seen_queries.add(query_hash)
        
        # Sort by combined score (vector similarity + AST similarity)
        combined_results.sort(
            key=lambda x: (x.get('score', 0) * 0.6 + x.get('ast_similarity', 0) * 0.4),
            reverse=True
        )
        
        return combined_results[:20]  # Return top 20 most relevant

    def _determine_optimization_strategy(self, query: SQLQuery,
                                      similar_queries: List[Dict]) -> List[OptimizationType]:
        """Determine which optimization types to apply based on query characteristics"""
        optimization_types = []
        
        # Always start with rule-based optimization
        optimization_types.append(OptimizationType.LLM_R2_REWRITE)
        
        # Add CodeLlama for complex queries
        if (query.complexity_score > 15 or
            query.join_count > 2 or
            query.ast_features.get('subquery_count', 0) > 1):
            optimization_types.append(OptimizationType.CODELLAMA_REFINEMENT)
        
        # Add CodeLlama if similar successful optimizations exist
        successful_similar = [
            q for q in similar_queries
            if q['payload'].get('confidence_score', 0) > 0.8
        ]
        if len(successful_similar) >= 3:
            if OptimizationType.CODELLAMA_REFINEMENT not in optimization_types:
                optimization_types.append(OptimizationType.CODELLAMA_REFINEMENT)
                
        return optimization_types

    async def batch_optimize(self, queries: List[Tuple[str, str]],
                          max_concurrent: int = 5) -> List[List[OptimizationResult]]:
        """Optimize multiple queries concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def optimize_single(sql_dialect_pair):
            sql, dialect = sql_dialect_pair
            async with semaphore:
                return await self.optimize_sql(sql, dialect)
                
        tasks = [optimize_single(pair) for pair in queries]
        return await asyncio.gather(*tasks)

# # --- EXAMPLE USAGE ---

# if __name__ == "__main__":
#     config = {
#         "codellama": {"model_name": "codellama/CodeLlama-34b-Instruct-hf"},
#         "use_cache": False
#     }
#     optimizer = ComprehensiveSQLOptimizer(config)
#     sql = "SELECT * FROM users WHERE age > 30"
#     results = asyncio.run(optimizer.optimize_sql(sql))
#     for result in results:
#         print("Optimized SQL:", result.optimized_sql)
#         print("Confidence:", result.confidence_score)
#         print("Explanation:", result.explanation)
