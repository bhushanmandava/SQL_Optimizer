# SQL_Optimizer# SQL Optimization System

A modular, production-ready system for advanced SQL query analysis, optimization, and semantic search. This project leverages state-of-the-art NLP models, graph-based features, and dual vector storage (Qdrant & Weaviate) to provide fast, intelligent SQL optimization and retrieval.

---

## Features

- **SQL Normalization:** Cleans and formats SQL queries, removing comments and standardizing identifiers.
- **AST Feature Extraction:** Analyzes queries to extract rich structural features (joins, subqueries, aggregates, etc.).
- **AST Shape Classification:** Categorizes queries into patterns (e.g., simple select, join-heavy, subquery-complex) for targeted optimization.
- **Hybrid Embedding Generation:** Combines CodeBERT, T5-SQL, and graph-based features for robust semantic vector representations.
- **Dual Vector Store:** Stores embeddings in Qdrant for fast similarity search and Weaviate for advanced metadata filtering.
- **Optimization Metadata:** Tracks optimization type, confidence, performance gain, suggested indexes, and more.
- **Prometheus Metrics:** Exposes counters, gauges, and histograms for monitoring optimization requests, cache hits, accuracy, and latency.

---

## Architecture

- **SQLGlot** for SQL parsing and AST analysis.
- **Transformers (CodeBERT, T5-SQL)** for semantic embeddings.
- **Qdrant** for approximate nearest neighbor (ANN) vector search.
- **Weaviate** for hybrid vector and metadata search.
- **Redis** for caching.
- **Prometheus** for metrics and monitoring.

---

## Installation

```bash
git clone https://github.com/your-org/sql-optimization-system.git
cd sql-optimization-system
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- torch
- transformers
- sqlglot
- qdrant-client
- weaviate-client
- redis
- prometheus-client
- numpy
- pyyaml

---

## Configuration

Create a `config.yaml` file in the project root:

```yaml
vector_stores:
  qdrant:
    host: localhost
    port: 6333
    collection: sql_queries
  weaviate:
    url: http://localhost:8080
    class: SqlQuery

embedding:
  codebert_weight: 0.4
  t5_weight: 0.3
  graph_weight: 0.3
```

---

## Usage

### As a Library

```python
from codebase import EnhancedSQLGlotAnalyzer, HybridEmbeddingGenerator, DualVectorStore, SQLQuery, OptimizationResult

# Initialize components
analyzer = EnhancedSQLGlotAnalyzer()
embedder = HybridEmbeddingGenerator(config)
vector_store = DualVectorStore(qdrant_config, weaviate_config)

# Analyze and optimize a SQL query
sql = "SELECT * FROM users WHERE age > 30"
normalized_sql = analyzer.normalize_sql(sql)
features = analyzer.extract_ast_features(normalized_sql)
shape = analyzer.classify_ast_shape(features)
embedding = embedder.generate_embedding(normalized_sql, features)
query_hash = hashlib.sha256(sql.encode()).hexdigest()

query = SQLQuery(
    original_sql=sql,
    query_hash=query_hash,
    ast_features=features,
    ast_shape=shape,
    embedding=embedding,
    normalized_sql=normalized_sql
)

# OptimizationResult would be produced by your optimization logic
result = OptimizationResult(
    optimized_sql="SELECT * FROM users WHERE age > 30",
    optimization_type="SYNTACTIC",
    confidence_score=1.0,
    explanation="No optimization needed."
)

vector_store.store_query(query, result)
```

### As a Service

You can build an async loop or REST API on top of this core to process incoming SQL queries, optimize them, and store results.

---

## Metrics

Prometheus metrics are exposed for:
- `sql_optimization_requests_total`
- `sql_cache_hits_total`
- `sql_optimization_accuracy`
- `sql_optimization_duration_seconds`
- `llm_inference_duration_seconds`

Start the Prometheus HTTP server in your main application:

```python
from prometheus_client import start_http_server
start_http_server(8000)
```

---

## Extending

- Add new optimization strategies by extending `OptimizationType` and implementing new logic.
- Integrate additional vector stores or embedding models as needed.
- Customize AST feature extraction for your SQL dialect or workload.


