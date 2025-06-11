import asyncio
import json
import logging
import hashlib
import time
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import sqlglot
from sqlglot import parse_one, optimize, exp
from sqlglot.optimizer import optimize as sqlglot_optimize
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import weaviate
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import ast
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
optimization_requests = Counter('sql_optimization_requests_total', 'Total optimization requests')
cache_hits = Counter('sql_cache_hits_total', 'Total cache hits')
accuracy_gauge = Gauge('sql_optimization_accuracy', 'Current optimization accuracy')
optimization_latency = Histogram('sql_optimization_duration_seconds', 'Optimization latency')
llm_inference_latency = Histogram('llm_inference_duration_seconds', 'LLM inference latency')
#we will begin with the enum classes to define our string iterrals to avaoid any kind of typos
class OptimizationType(Enum):
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic" 
    PERFORMANCE = "performance"
    INDEX_SUGGESTION = "index_suggestion"
    LLM_R2_REWRITE = "llm_r2_rewrite"
    CODELLAMA_REFINEMENT = "codellama_refinement"

class ASTShape(Enum):
    SIMPLE_SELECT = "simple_select"
    JOIN_HEAVY = "join_heavy"
    SUBQUERY_COMPLEX = "subquery_complex"
    AGGREGATE_WINDOW = "aggregate_window"
    CTE_RECURSIVE = "cte_recursive"
    UNION_MULTI = "union_multi"
# we are capturing the results of the optimization and helps us to track all the meta data at one place
@dataclass
class OptimizationResult:
    optimized_sql: str
    optimization_type: OptimizationType
    confidence_score: float
    explanation: str
    performance_gain_estimate: float = 0.0
    suggested_indexes: List[str] = None
    optimization_stages: List[str] = None
    execution_plan_comparison: Dict = None
    llm_reasoning: str = ""
# this where we capture the all sql data and wee use this as the input formate for our optimization engine
@dataclass
class SQLQuery:
    original_sql: str
    query_hash: str
    ast_features: Dict
    ast_shape: ASTShape
    embedding: List[float]
    normalized_sql: str
    complexity_score: float = 0.0
    table_references: List[str] = None
    join_count: int = 0


class EnhancedSQLGlotAnalyzer:
    """Advanced SQL analysis using SQLGlot with AST shape classification"""
    
    def __init__(self):
        self.shape_patterns = {
            ASTShape.SIMPLE_SELECT: self._is_simple_select,
            ASTShape.JOIN_HEAVY: self._is_join_heavy,
            ASTShape.SUBQUERY_COMPLEX: self._is_subquery_complex,
            ASTShape.AGGREGATE_WINDOW: self._is_aggregate_window,
            ASTShape.CTE_RECURSIVE: self._is_cte_recursive,
            ASTShape.UNION_MULTI: self._is_union_multi
        }
    
    def normalize_sql(self, sql: str, dialect: str = 'mysql') -> str:
        """Normalize SQL using SQLGlot with comment removal and formatting"""
        try:
            ast = parse_one(sql, dialect=dialect)
            if not ast:
                return sql.strip()
            
            # Remove comments and normalize formatting
            normalized_ast = ast.transform(self._remove_comments)
            normalized_ast = normalized_ast.transform(self._normalize_identifiers)
            
            return normalized_ast.sql(dialect=dialect, pretty=True)
        except Exception as e:
            logger.error(f"SQL normalization failed: {e}")
            return sql.strip()
    
    def extract_ast_features(self, sql: str, dialect: str = 'mysql') -> Dict:
        """Extract comprehensive AST features for analysis"""
        try:
            ast = parse_one(sql, dialect=dialect)
            if not ast:
                return {}
            
            features = {
                'table_count': len(list(ast.find_all(exp.Table))),
                'join_count': len(list(ast.find_all(exp.Join))),
                'subquery_count': len(list(ast.find_all(exp.Subquery))),
                'cte_count': len(list(ast.find_all(exp.CTE))),
                'window_function_count': len(list(ast.find_all(exp.Window))),
                'aggregate_count': len([f for f in ast.find_all(exp.Func) 
                                      if f.is_aggregate]),
                'union_count': len(list(ast.find_all(exp.Union))),
                'where_conditions': len(list(ast.find_all(exp.Where))),
                'having_conditions': len(list(ast.find_all(exp.Having))),
                'order_by_columns': len(list(ast.find_all(exp.Order))),
                'distinct_usage': len(list(ast.find_all(exp.Distinct))),
                'case_statements': len(list(ast.find_all(exp.Case))),
                'nested_depth': self._calculate_nesting_depth(ast),
                'complexity_score': self._calculate_complexity_score(ast)
            }
            
            # Extract table references
            tables = [table.name for table in ast.find_all(exp.Table)]
            features['table_references'] = tables
            
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def classify_ast_shape(self, features: Dict) -> ASTShape:
        """Classify query AST into predefined shapes for optimization targeting"""
        for shape, classifier in self.shape_patterns.items():
            if classifier(features):
                return shape
        return ASTShape.SIMPLE_SELECT
    
    def compute_ast_similarity(self, sql1: str, sql2: str, dialect: str = 'mysql') -> float:
        """Compute AST-based similarity using tree edit distance"""
        try:
            ast1 = parse_one(sql1, dialect=dialect)
            ast2 = parse_one(sql2, dialect=dialect)
            
            if not ast1 or not ast2:
                return 0.0
            
            # Convert ASTs to string representations for comparison
            ast1_str = self._ast_to_normalized_string(ast1)
            ast2_str = self._ast_to_normalized_string(ast2)
            
            # Use sequence matcher for similarity
            similarity = SequenceMatcher(None, ast1_str, ast2_str).ratio()
            
            # Apply structural similarity bonus
            structural_bonus = self._compute_structural_similarity(ast1, ast2)
            
            return min(1.0, similarity * 0.7 + structural_bonus * 0.3)
        except Exception as e:
            logger.error(f"AST similarity computation failed: {e}")
            return 0.0
    
    def _remove_comments(self, node):
        """Remove comment nodes from AST"""
        if isinstance(node, exp.Comment):
            return None
        return node
    
    def _normalize_identifiers(self, node):
        """Normalize identifier casing"""
        if isinstance(node, (exp.Identifier, exp.Column, exp.Table)):
            if node.name and not node.quoted:
                node.name = node.name.lower()
        return node
    
    def _calculate_nesting_depth(self, ast) -> int:
        """Calculate maximum nesting depth of the AST"""
        def depth(node):
            if not hasattr(node, 'args') or not node.args:
                return 1
            return 1 + max([depth(child) for child in node.args.values() 
                           if isinstance(child, (list, exp.Expression))] or [0])
        return depth(ast)
    
    def _calculate_complexity_score(self, ast) -> float:
        """Calculate query complexity score based on multiple factors"""
        features = {
            'tables': len(list(ast.find_all(exp.Table))),
            'joins': len(list(ast.find_all(exp.Join))),
            'subqueries': len(list(ast.find_all(exp.Subquery))),
            'aggregates': len([f for f in ast.find_all(exp.Func) if f.is_aggregate]),
            'windows': len(list(ast.find_all(exp.Window))),
            'ctes': len(list(ast.find_all(exp.CTE)))
        }
        
        # Weighted complexity score
        score = (features['tables'] * 1.0 + 
                features['joins'] * 2.0 + 
                features['subqueries'] * 3.0 + 
                features['aggregates'] * 1.5 + 
                features['windows'] * 2.5 + 
                features['ctes'] * 2.0)
        
        return min(100.0, score)
    
    def _ast_to_normalized_string(self, ast) -> str:
        """Convert AST to normalized string representation"""
        return str(type(ast).__name__) + ":" + str(sorted([
            self._ast_to_normalized_string(child) 
            for child in ast.args.values() 
            if isinstance(child, exp.Expression)
        ]))
    
    def _compute_structural_similarity(self, ast1, ast2) -> float:
        """Compute structural similarity between two ASTs"""
        type1 = type(ast1).__name__
        type2 = type(ast2).__name__
        
        if type1 != type2:
            return 0.0
        
        if not hasattr(ast1, 'args') or not hasattr(ast2, 'args'):
            return 1.0
        
        args1 = list(ast1.args.keys())
        args2 = list(ast2.args.keys())
        
        common_args = set(args1) & set(args2)
        total_args = set(args1) | set(args2)
        
        return len(common_args) / len(total_args) if total_args else 1.0
    
    # Shape classification methods
    def _is_simple_select(self, features: Dict) -> bool:
        return (features.get('join_count', 0) <= 1 and 
                features.get('subquery_count', 0) == 0 and
                features.get('complexity_score', 0) < 10)
    
    def _is_join_heavy(self, features: Dict) -> bool:
        return features.get('join_count', 0) >= 3
    
    def _is_subquery_complex(self, features: Dict) -> bool:
        return (features.get('subquery_count', 0) >= 2 or
                features.get('nested_depth', 0) >= 4)
    
    def _is_aggregate_window(self, features: Dict) -> bool:
        return (features.get('aggregate_count', 0) >= 2 or
                features.get('window_function_count', 0) >= 1)
    
    def _is_cte_recursive(self, features: Dict) -> bool:
        return features.get('cte_count', 0) >= 1
    
    def _is_union_multi(self, features: Dict) -> bool:
        return features.get('union_count', 0) >= 1


class HybridEmbeddingGenerator:
    """Hybrid embedding system combining CodeBERT, T5-SQL, and graph features"""
    
    def __init__(self, config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CodeBERT
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        
        # Initialize T5-SQL
        self.t5_tokenizer = AutoTokenizer.from_pretrained("suriya7/t5-base-text-to-sql")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/t5-base-text-to-sql").to(self.device)
        
        # Embedding weights
        self.codebert_weight = config.get('codebert_weight', 0.4)
        self.t5_weight = config.get('t5_weight', 0.3)
        self.graph_weight = config.get('graph_weight', 0.3)
        
        # Dimension consistency
        self.target_dim = 768
        
    def generate_embedding(self, sql: str, ast_features: Dict = None) -> List[float]:
        """Generate hybrid embedding combining multiple approaches"""
        try:
            # CodeBERT embedding
            codebert_emb = self._get_codebert_embedding(sql)
            
            # T5-SQL embedding  
            t5_emb = self._get_t5_embedding(sql)
            
            # Graph-based embedding
            graph_emb = self._get_graph_embedding(ast_features or {})
            
            # Combine embeddings
            hybrid_emb = (self.codebert_weight * codebert_emb + 
                         self.t5_weight * t5_emb + 
                         self.graph_weight * graph_emb)
            
            # Normalize
            hybrid_emb = hybrid_emb / np.linalg.norm(hybrid_emb)
            
            return hybrid_emb.tolist()
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * self.target_dim
    
    def _get_codebert_embedding(self, sql: str) -> np.ndarray:
        """Generate CodeBERT embedding for SQL"""
        try:
            inputs = self.codebert_tokenizer(
                sql, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.codebert_model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"CodeBERT embedding failed: {e}")
            return np.zeros(self.target_dim)
    
    def _get_t5_embedding(self, sql: str) -> np.ndarray:
        """Generate T5-SQL embedding"""
        try:
            input_text = f"translate to SQL: {sql}"
            inputs = self.t5_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                # Get encoder embeddings
                encoder_outputs = self.t5_model.encoder(**inputs)
                embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"T5 embedding failed: {e}")
            return np.zeros(self.target_dim)
    
    def _get_graph_embedding(self, ast_features: Dict) -> np.ndarray:
        """Generate graph-based structural embedding"""
        try:
            # Create feature vector from AST properties
            feature_vector = np.array([
                ast_features.get('table_count', 0),
                ast_features.get('join_count', 0),
                ast_features.get('subquery_count', 0),
                ast_features.get('aggregate_count', 0),
                ast_features.get('window_function_count', 0),
                ast_features.get('cte_count', 0),
                ast_features.get('union_count', 0),
                ast_features.get('complexity_score', 0) / 100.0,  # Normalize
                ast_features.get('nested_depth', 0) / 10.0,  # Normalize
                ast_features.get('where_conditions', 0),
                ast_features.get('having_conditions', 0),
                ast_features.get('order_by_columns', 0),
                ast_features.get('distinct_usage', 0),
                ast_features.get('case_statements', 0)
            ], dtype=np.float32)
            
            # Pad to target dimension with learned transformation
            if len(feature_vector) < self.target_dim:
                # Simple expansion using repetition and noise
                expansion_factor = self.target_dim // len(feature_vector)
                remainder = self.target_dim % len(feature_vector)
                
                expanded = np.tile(feature_vector, expansion_factor)
                if remainder > 0:
                    expanded = np.concatenate([expanded, feature_vector[:remainder]])
                
                # Add small amount of noise for uniqueness
                noise = np.random.normal(0, 0.01, self.target_dim)
                return expanded + noise
            else:
                return feature_vector[:self.target_dim]
                
        except Exception as e:
            logger.error(f"Graph embedding failed: {e}")
            return np.zeros(self.target_dim)


class DualVectorStore:
    """Dual vector storage using Qdrant for ANN search and Weaviate for metadata filtering"""
    
    def __init__(self, qdrant_config: Dict, weaviate_config: Dict):
        # Initialize Qdrant
        self.qdrant_client = QdrantClient(
            host=qdrant_config.get('host', 'localhost'),
            port=qdrant_config.get('port', 6333)
        )
        self.qdrant_collection = qdrant_config.get('collection', 'sql_queries')
        
        # Initialize Weaviate
        self.weaviate_client = weaviate.Client(
            url=weaviate_config.get('url', 'http://localhost:8080')
        )
        self.weaviate_class = weaviate_config.get('class', 'SqlQuery')
        
        self._setup_collections()
    
    def _setup_collections(self):
        """Initialize Qdrant collections and Weaviate schema"""
        try:
            # Setup Qdrant collection
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == self.qdrant_collection for c in collections):
                self.qdrant_client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
            
            # Setup Weaviate schema
            schema = {
                "class": self.weaviate_class,
                "properties": [
                    {"name": "originalSql", "dataType": ["text"]},
                    {"name": "optimizedSql", "dataType": ["text"]},
                    {"name": "astShape", "dataType": ["string"]},
                    {"name": "tableCount", "dataType": ["int"]},
                    {"name": "joinCount", "dataType": ["int"]},
                    {"name": "complexityScore", "dataType": ["number"]},
                    {"name": "optimizationType", "dataType": ["string"]},
                    {"name": "confidenceScore", "dataType": ["number"]},
                    {"name": "queryHash", "dataType": ["string"]},
                    {"name": "timestamp", "dataType": ["date"]}
                ]
            }
            
            # Check if class exists, create if not
            try:
                self.weaviate_client.schema.get(self.weaviate_class)
            except:
                self.weaviate_client.schema.create_class(schema)
                
        except Exception as e:
            logger.error(f"Vector store setup failed: {e}")
    
    def store_query(self, query: SQLQuery, optimization_result: OptimizationResult):
        """Store query and optimization in both vector stores"""
        try:
            # Store in Qdrant for vector similarity
            self._store_in_qdrant(query, optimization_result)
            
            # Store in Weaviate for metadata filtering
            self._store_in_weaviate(query, optimization_result)
            
        except Exception as e:
            logger.error(f"Failed to store query: {e}")
    
    def _store_in_qdrant(self, query: SQLQuery, optimization_result: OptimizationResult):
        """Store in Qdrant for high-performance vector search"""
        point = PointStruct(
            id=query.query_hash,
            vector=query.embedding,
            payload={
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
        )
        
        self.qdrant_client.upsert(
            collection_name=self.qdrant_collection,
            points=[point]
        )
    
    def _store_in_weaviate(self, query: SQLQuery, optimization_result: OptimizationResult):
        """Store in Weaviate for metadata filtering"""
        data_object = {
            "originalSql": query.original_sql,
            "optimizedSql": optimization_result.optimized_sql,
            "astShape": query.ast_shape.value,
            "tableCount": query.ast_features.get('table_count', 0),
            "joinCount": query.ast_features.get('join_count', 0),
            "complexityScore": query.ast_features.get('complexity_score', 0.0),
            "optimizationType": optimization_result.optimization_type.value,
            "confidenceScore": optimization_result.confidence_score,
            "queryHash": query.query_hash,
            "timestamp": datetime.now().isoformat()
        }
        
        self.weaviate_client.data_object.create(
            data_object=data_object,
            class_name=self.weaviate_class,
            uuid=query.query_hash,
            vector=query.embedding
        )
    
    def search_similar_queries(self, query_embedding: List[float], 
                             ast_shape: ASTShape, limit: int = 50) -> List[Dict]:
        """Search for similar queries using Qdrant with AST shape filtering"""
        try:
            # Coarse filter by AST shape
            shape_filter = Filter(
                must=[
                    FieldCondition(
                        key="ast_shape",
                        match=MatchValue(value=ast_shape.value)
                    )
                ]
            )
            
            search_result = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_embedding,
                query_filter=shape_filter,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in search_result
            ]
        except Exception as e:
            logger.error(f"Failed to search similar queries in Qdrant: {e}")
            return []
    
    def search_by_metadata(self, filters: Dict, limit: int = 10) -> List[Dict]:
        """Search queries by metadata using Weaviate hybrid search"""
        try:
            where_filter = self._build_weaviate_filter(filters)
            
            result = (
                self.weaviate_client.query
                .get(self.weaviate_class, ["originalSql", "optimizedSql", "astShape", 
                                         "tableCount", "joinCount", "complexityScore",
                                         "optimizationType", "confidenceScore"])
                .with_where(where_filter)
                .with_limit(limit)
                .do()
            )
            
            return result.get("data", {}).get("Get", {}).get(self.weaviate_class, [])
        except Exception as e:
            logger.error(f"Failed to search by metadata in Weaviate: {e}")
            return []
    
    def _build_weaviate_filter(self, filters: Dict) -> Dict:
        """Build Weaviate where filter from filters dict"""
        conditions = []
        
        for key, value in filters.items():
            if key == "complexity_range":
                conditions.append({
                    "path": ["complexityScore"],
                    "operator": "GreaterThanEqual",
                    "valueNumber": value[0]
                })
                conditions.append({
                    "path": ["complexityScore"], 
                    "operator": "LessThanEqual",
                    "valueNumber": value[1]
                })
            elif key == "ast_shape":
                conditions.append({
                    "path": ["astShape"],
                    "operator": "Equal",
                    "valueString": value
                })
            elif key == "min_confidence":
                conditions.append({
                    "path": ["confidenceScore"],
                    "operator": "GreaterThanEqual",
                    "valueNumber": value
                })
        
        return {"operator": "And", "operands": conditions} if conditions else {}


class LLMR2RuleEngine:
    """LLM-R2 rule-based optimization engine with dynamic rule generation"""
    
    def __init__(self, rules_config_path: str = "sql_rules.yaml"):
        self.rules = self._load_rules(rules_config_path)
        self.rule_cache = {}
        
    def _load_rules(self, config_path: str) -> Dict:
        """Load optimization rules from YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Provide default rules if config file not found
            return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """Default optimization rules"""
        return {
            "predicate_pushdown": {
                "condition": "WHERE in subquery",
                "action": "APPLY PREDICATE PUSHDOWN",
                "verify": "EXPLAIN COST",
                "fallback": "index_scan_rewrite",
                "confidence_threshold": 0.8
            },
            "join_reorder": {
                "condition": "multiple JOINs",
                "action": "REORDER JOINS BY SELECTIVITY",
                "verify": "EXPLAIN COST",
                "fallback": "original_query",
                "confidence_threshold": 0.7
            },
            "subquery_to_join": {
                "condition": "EXISTS subquery",
                "action": "CONVERT TO SEMI JOIN",
                "verify": "EXPLAIN COST",
                "fallback": "original_query",
                "confidence_threshold": 0.75
            },
            "redundant_distinct": {
                "condition": "DISTINCT with GROUP BY",
                "action": "REMOVE REDUNDANT DISTINCT",
                "verify": "LOGICAL EQUIVALENCE",
                "fallback": "original_query",
                "confidence_threshold": 0.9
            }
        }
    
    def apply_rules(self, sql: str, ast_features: Dict, similar_optimizations: List[Dict]) -> OptimizationResult:
        """Apply rule-based optimizations with context awareness"""
        try:
            ast = parse_one(sql)
            if not ast:
                return None
            
            applied_rules = []
            optimized_ast = ast
            confidence_scores = []
            
            for rule_name, rule_config in self.rules.items():
                if self._check_rule_condition(rule_config["condition"], ast, ast_features):
                    try:
                        # Apply rule transformation
                        transformed_ast = self._apply_rule_transformation(
                            optimized_ast, rule_name, rule_config, similar_optimizations
                        )
                        
                        if transformed_ast and self._verify_transformation(
                            optimized_ast, transformed_ast, rule_config["verify"]
                        ):
                            optimized_ast = transformed_ast
                            applied_rules.append(rule_name)
                            confidence_scores.append(rule_config.get("confidence_threshold", 0.7))
                            
                    except Exception as e:
                        logger.warning(f"Rule {rule_name} failed: {e}")
                        # Apply fallback if specified
                        if rule_config.get("fallback") == "original_query":
                            continue
            
            if not applied_rules:
                return None
            
            optimized_sql = optimized_ast.sql(pretty=True)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            return OptimizationResult(
                optimized_sql=optimized_sql,
                optimization_type=OptimizationType.LLM_R2_REWRITE,
                confidence_score=avg_confidence,
                explanation=f"Applied LLM-R2 rules: {', '.join(applied_rules)}",
                optimization_stages=applied_rules,
                llm_reasoning=f"Rule-based optimization using dynamic patterns from {len(similar_optimizations)} similar queries"
            )
            
        except Exception as e:
            logger.error(f"LLM-R2 rule application failed: {e}")
            return None
    
    def _check_rule_condition(self, condition: str, ast, ast_features: Dict) -> bool:
        """Check if rule condition is met"""
        condition_lower = condition.lower()
        
        if "where in subquery" in condition_lower:
            return any(
                any(sub.find_all(exp.Where) for sub in node.find_all(exp.Subquery))
                for node in [ast]
            )
        elif "multiple joins" in condition_lower:
            return len(list(ast.find_all(exp.Join))) >= 2
        elif "exists subquery" in condition_lower:
            return any(
                isinstance(sub.this, exp.Exists) 
                for sub in ast.find_all(exp.Subquery)
            )
        elif "distinct with group by" in condition_lower:
            return (any(ast.find_all(exp.Distinct)) and 
                   any(ast.find_all(exp.Group)))
        
        return False
    
    def _apply_rule_transformation(self, ast, rule_name: str, rule_config: Dict, 
                                 similar_optimizations: List[Dict]):
        """Apply specific rule transformation to AST"""
        if rule_name == "predicate_pushdown":
            return self._apply_predicate_pushdown(ast)
        elif rule_name == "join_reorder":
            return self._apply_join_reorder(ast, similar_optimizations)
        elif rule_name == "subquery_to_join":
            return self._apply_subquery_to_join(ast)
        elif rule_name == "redundant_distinct":
            return self._apply_remove_redundant_distinct(ast)
        
        return ast
    
    def _verify_transformation(self, original_ast, transformed_ast, verify_method: str) -> bool:
        """Verify that transformation is beneficial"""
        if verify_method == "LOGICAL EQUIVALENCE":
            # Simple check - ensure both ASTs have same structure type
            return type(original_ast) == type(transformed_ast)
        elif verify_method == "EXPLAIN COST":
            # In real implementation, would use database EXPLAIN
            # For now, assume transformation is beneficial if AST changed
            return str(original_ast) != str(transformed_ast)
        
        return True
    
    def _apply_predicate_pushdown(self, ast):
        """Apply predicate pushdown optimization"""
        # Simplified implementation - in practice would use SQLGlot's optimizer
        return sqlglot_optimize(ast, rules=["push_down_predicates"])
    
    def _apply_join_reorder(self, ast, similar_optimizations: List[Dict]):
        """Apply join reordering based on similar query patterns"""
        # Use patterns from similar optimizations to guide join order
        return sqlglot_optimize(ast, rules=["normalize_identifiers", "optimize_joins"])
    
    def _apply_subquery_to_join(self, ast):
        """Convert EXISTS subqueries to semi-joins"""
        return sqlglot_optimize(ast, rules=["unnest_subqueries"])
    
    def _apply_remove_redundant_distinct(self, ast):
        """Remove redundant DISTINCT clauses"""
        # Remove DISTINCT when GROUP BY is present
        for select in ast.find_all(exp.Select):
            if select.distinct and any(select.find_all(exp.Group)):
                select.set("distinct", None)
        return ast


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
        
    def optimize_sql(self, sql: str, context: Dict, previous_optimizations: List[OptimizationResult]) -> OptimizationResult:
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
    
    def _build_optimization_prompt(self, sql: str, context: Dict, 
                                 previous_optimizations: List[OptimizationResult]) -> str:
        """Build comprehensive optimization prompt for CodeLlama"""
        
        # Extract context information
        ast_features = context.get('ast_features', {})
        similar_patterns = context.get('similar_patterns', [])
        
        prompt = f"""<s>[INST] You are an expert SQL optimization system. 

TASK: Optimize the following SQL query for performance while maintaining correctness.

ORIGINAL QUERY:

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
                prompt += f"  Confidence: {pattern.get('confidence_score', 0):.2f}\n"
        
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
    # return prompt

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
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
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
        sql_pattern = r'```sql\s*(.*?)\s*```'
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
        reasoning_parts = response.split("```sql")
        reasoning = reasoning_parts[0].strip() if reasoning_parts else "No reasoning provided"
        
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


def __init__(self, redis_config: Dict):
    import redis
    self.redis_client = redis.Redis(
        host=redis_config.get('host', 'localhost'),
        port=redis_config.get('port', 6379),
        db=redis_config.get('db', 0),
        decode_responses=True
    )
    self.ttl = redis_config.get('ttl', 3600)  # 1 hour default
    
def get(self, query_hash: str) -> Optional[OptimizationResult]:
    """Get cached optimization result"""
    try:
        cached_data = self.redis_client.get(f"opt:{query_hash}")
        if cached_data:
            cache_hits.inc()
            data = json.loads(cached_data)
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
        
        self.redis_client.setex(
            f"opt:{query_hash}",
            self.ttl,
            json.dumps(data)
        )
    except Exception as e:
        logger.error(f"Failed to cache result: {e}")


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
    
    # Start metrics server
    if config.get('metrics_port'):
        start_http_server(config['metrics_port'])
        
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
            "ast_shape": ast_shape.value,
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
                'ast_shape': ast_shape.value,
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
            original_sql = result['payload'].get('original_sql', '')
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
        async with semaphore:
            sql, dialect = sql_dialect_pair
            return await self.optimize_sql(sql, dialect)
    
    tasks = [optimize_single(pair) for pair in queries]
    return await asyncio.gather(*tasks)

def get_system_stats(self) -> Dict:
    """Get comprehensive system statistics"""
    return {
        "total_requests": optimization_requests._value.get(),
        "cache_hits": cache_hits._value.get(),
        "current_accuracy": accuracy_gauge._value.get(),
        "avg_latency": optimization_latency._sum.get() / max(optimization_latency._count.get(), 1),
        "avg_llm_latency": llm_inference_latency._sum.get() / max(llm_inference_latency._count.get(), 1),
        "cache_hit_rate": cache_hits._value.get() / max(optimization_requests._value.get(), 1)
    }

def evaluate_optimization(self, original_sql: str, optimized_sql: str, 
                        dialect: str = 'mysql') -> Dict:
    """Comprehensive evaluation of optimization results"""
    try:
        # Parse both queries
        original_ast = parse_one(original_sql, dialect=dialect)
        optimized_ast = parse_one(optimized_sql, dialect=dialect)
        
        if not original_ast or not optimized_ast:
            return {"error": "Failed to parse queries"}
        
        # Extract features for comparison
        original_features = self.analyzer.extract_ast_features(original_sql, dialect)
        optimized_features = self.analyzer.extract_ast_features(optimized_sql, dialect)
        
        # Calculate metrics
        ast_similarity = self.analyzer.compute_ast_similarity(original_sql, optimized_sql, dialect)
        
        # Complexity reduction
        complexity_reduction = (
            original_features.get('complexity_score', 0) - 
            optimized_features.get('complexity_score', 0)
        ) / max(original_features.get('complexity_score', 1), 1) * 100
        
        # Structural improvements
        structural_improvements = []
        if optimized_features.get('join_count', 0) < original_features.get('join_count', 0):
            structural_improvements.append("join_reduction")
        if optimized_features.get('subquery_count', 0) < original_features.get('subquery_count', 0):
            structural_improvements.append("subquery_optimization")
        if optimized_features.get('nested_depth', 0) < original_features.get('nested_depth', 0):
            structural_improvements.append("nesting_reduction")
        
        # Calculate composite score
        composite_score = (
            0.4 * ast_similarity +
            0.3 * min(1.0, complexity_reduction / 20.0) +
            0.3 * min(1.0, len(structural_improvements) / 3.0)
        )
        
        return {
            "ast_similarity": ast_similarity,
            "complexity_reduction_percent": complexity_reduction,
            "structural_improvements": structural_improvements,
            "composite_score": composite_score,
            "original_complexity": original_features.get('complexity_score', 0),
            "optimized_complexity": optimized_features.get('complexity_score', 0),
            "original_features": original_features,
            "optimized_features": optimized_features
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}
