# Comprehensive SQL Query Optimizer Agent Instructions

## 1. Core Optimization Objectives

The SQL Query Optimizer Agent should be designed to transform unoptimized SQL queries into their most efficient form, with the primary goal of reducing time complexity\[1]\[2]. The agent should analyze incoming queries, identify inefficiencies, and apply a series of optimization techniques to produce queries that execute faster while maintaining semantic equivalence\[3]\[4].

The optimizer must focus on minimizing computational resources required for query execution by implementing both rule-based and cost-based optimization strategies\[5]\[6]. Time complexity reduction should be achieved through systematic transformation of the query structure, execution plan, and access methods\[7]\[8].

## 2. Query Analysis Framework

### 2.1 Query Complexity Assessment

Before optimization, the agent should evaluate the incoming query's complexity using metrics such as:

* Number of tables and joins involved in the query\[9]
* Number of columns referenced and returned\[9]\[10]
* Types and quantity of operators and expressions used\[9]\[10]
* Presence of subqueries, CTEs, and complex clauses (GROUP BY, HAVING, etc.)\[10]
* Estimated cardinality (number of rows) to be processed\[6]\[9]

The agent should calculate a complexity score to prioritize optimization efforts, focusing more resources on queries with higher complexity scores that would benefit most from optimization\[9]\[10].

### 2.2 Execution Plan Analysis

The agent must analyze the query's execution plan to identify bottlenecks and inefficiencies\[8]\[11]:

* Generate and examine the estimated execution plan\[5]\[11]
* Identify high-cost operations that consume excessive resources\[8]
* Evaluate index usage patterns and opportunities\[8]\[11]
* Check for cardinality estimate accuracy and potential mismatches\[8]\[6]
* Analyze the execution order of operations\[8]\[12]

## 3. Rule-Based Optimization Techniques

### 3.1 Predicate Optimization

* Push down filter conditions as early as possible in the execution plan\[7]\[12]
* Simplify complex WHERE conditions by applying logical transformations\[3]\[4]
* Rewrite predicates to leverage available indexes effectively\[2]\[13]
* Avoid functions on indexed columns in WHERE clauses\[1]\[3]
* Optimize date range filters instead of using date functions\[1]\[2]
* Rewrite OR conditions with the same column into IN or ANY operators\[14]
* Rewrite AND conditions with the same column into ALL operators\[14]

### 3.2 Join Optimization

* Select the most appropriate join type based on table sizes and cardinality\[3]\[7]
* Reorder joins to process smaller result sets first\[7]\[5]
* Replace nested loops with hash or merge joins when appropriate\[3]\[7]
* Eliminate redundant joins\[15]
* Convert subqueries to joins when beneficial\[3]\[4]
* Apply join predicate pushdown\[16]\[12]
* Use join hints when the optimizer consistently makes poor choices\[13]

### 3.3 Subquery Transformation

* Flatten nested subqueries when possible\[7]\[4]
* Convert correlated subqueries to joins\[17]\[4]
* Unnest subqueries in predicates into joins\[3]\[17]
* Apply subquery decorrelation techniques\[4]
* Convert EXISTS subqueries to semi-joins\[17]\[4]
* Transform IN subqueries to joins with distinct operations\[17]\[4]

### 3.4 Query Rewriting

* Eliminate unnecessary DISTINCT operations\[13]\[15]
* Replace SELECT \* with specific column lists\[1]\[2]\[13]
* Rewrite complex expressions to simpler equivalent forms\[3]\[4]
* Convert UNION to UNION ALL when duplicates are irrelevant\[2]\[7]
* Merge views and derived tables into the main query when beneficial\[16]\[15]
* Eliminate common subexpressions\[12]
* Convert OR expansions to UNION ALL when advantageous\[17]

## 4. Cost-Based Optimization Strategies

### 4.1 Statistics Utilization

* Ensure statistics are current for all tables\[7]\[6]
* Use column distribution stats to estimate selectivity\[5]\[6]
* Consider data skew in join cardinality\[5]\[6]
* Use multi-column stats for correlated columns\[6]
* Analyze histogram data for range predicate cost estimation\[5]\[6]

### 4.2 Access Path Selection

* Choose between full table scan and index access based on selectivity\[2]\[5]
* Consider covering indexes\[2]\[13]
* Evaluate index intersection\[2]\[13]
* Choose between clustered and non-clustered index access\[13]\[5]
* Use index-only access to avoid table lookups\[2]\[13]

### 4.3 Join Strategy Selection

* Choose optimal join algorithm: nested loops, hash, or merge joins\[3]\[7]
* Select build and probe sides for hash joins\[5]
* Use sort-merge joins for pre-sorted data\[5]
* Decide between broadcast and shuffle joins in distributed systems\[6]
* Tune join buffer size based on memory\[7]

## 5. Advanced Optimization Techniques

### 5.1 Materialized View Utilization

* Rewrite queries to use materialized views\[7]
* Suggest materialized view creation for repeated patterns\[7]
* Use incremental materialized view maintenance\[7]

### 5.2 Partition Pruning

* Enable partition elimination using partition keys\[7]\[6]
* Rewrite queries for partition-wise joins\[7]
* Support dynamic partition pruning\[7]\[6]

### 5.3 Parallel Execution

* Identify operations eligible for parallel execution\[7]\[18]
* Balance degree of parallelism to avoid contention\[7]
* Distribute data for parallel joins\[7]\[6]
* Optimize parallel aggregation strategies\[7]

### 5.4 Memory Optimization

* Adjust memory grant size based on estimates\[7]
* Reduce disk spilling for memory-heavy ops\[7]
* Use memory-optimized execution plans\[7]

## 6. Query Transformation Workflow

1. Parse SQL into AST\[19]
2. Perform syntactic validation\[16]
3. Transform AST into logical query plan\[16]
4. Apply rule-based transformations\[16]\[12]
5. Generate alternate plans\[5]\[16]
6. Estimate cost for each plan\[5]\[6]
7. Select plan with lowest cost\[5]\[6]
8. Convert final plan into optimized SQL\[16]
9. Provide optimized SQL with detailed explanation\[8]

## 7. Time Complexity Considerations

* Reduce O(n^2) to O(n log n) or O(n) where feasible\[18]
* Prefer index lookups (O(log n)) over table scans (O(n))\[18]
* Replace nested loops (O(n×m)) with hash joins (O(n+m))\[18]
* Minimize sort operations (O(n log n))\[18]
* Assess data volume impact on complexity\[18]

## 8. Output Format and Explanations

1. Original query with inefficiencies highlighted\[8]
2. Optimized query with clear formatting\[8]
3. Summary of transformations and their rationale\[8]
4. Estimated performance improvements\[8]
5. Key optimization decision rationale\[8]
6. Schema or index improvement recommendations\[2]\[13]

## 9. Continuous Learning and Improvement

1. Track real-world query performance\[7]
2. Compare actual vs. estimated metrics\[7]\[8]
3. Refine cost models using real data\[7]\[6]
4. Adapt based on query and schema patterns\[7]\[6]
5. Integrate new optimization strategies over time\[7]

## 10. Implementation Considerations

* Support multiple SQL dialects\[19]
* Integrate with monitoring/performance tools\[7]\[8]
* Handle vendor-specific optimization hints\[13]
* Balance optimization time with gains\[5]\[6]
* Support both automated and guided modes\[8]
* Preserve semantic equivalence\[3]\[4]
