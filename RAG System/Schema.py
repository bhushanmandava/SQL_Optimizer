import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
from prometheus_client import Counter, Gauge, Histogram

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Prometheus metrics
optimization_requests = Counter('sql_optimization_requests_total', 'Total optimization requests')
cache_hits = Counter('sql_cache_hits_total', 'Total cache hits')
accuracy_gauge = Gauge('sql_optimization_accuracy', 'Current optimization accuracy')
optimization_latency = Histogram('sql_optimization_duration_seconds', 'Optimization latency')
llm_inference_latency = Histogram('llm_inference_duration_seconds', 'LLM inference latency')


# Enum to define types of optimization
class OptimizationType(Enum):
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"
    INDEX_SUGGESTION = "index_suggestion"
    LLM_R2_REWRITE = "llm_r2_rewrite"
    CODELLAMA_REFINEMENT = "codellama_refinement"


# Enum to describe AST shape categories
class ASTShape(Enum):
    SIMPLE_SELECT = "simple_select"
    JOIN_HEAVY = "join_heavy"
    SUBQUERY_COMPLEX = "subquery_complex"
    AGGREGATE_WINDOW = "aggregate_window"
    CTE_RECURSIVE = "cte_recursive"
    UNION_MULTI = "union_multi"


# Class to capture optimization result
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


# Class to represent the SQL query structure
@dataclass
class SQLQuery:
    # original_sql: str
    query_hash: str
    ast_features: Dict
    ast_shape: ASTShape
    embedding: List[float]
    normalized_sql: str
    complexity_score: float = 0.0
    table_references: List[str] = None
    join_count: int = 0
