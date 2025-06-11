import yaml
import logging
from typing import Dict, List
from sqlglot import parse_one, exp
from sqlglot.optimizer import optimize as sqlglot_optimize
from sqlglot.optimizer import optimize_joins
from Schema import OptimizationResult, OptimizationType, logger
# import loggingclear

# logging.basicConfig(level=logging.DEBUG)
class LLMR2RuleEngine:
    def __init__(self, rules_config_path: str = "sql_rules.yaml"):
        self.rules = self._load_rules(rules_config_path)
        self.rule_cache = {}
        logger.debug("[LOG] Rule engine initialized with rules: %s", self.rules)

    def _load_rules(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                rules = yaml.safe_load(f)
                logger.debug("[LOG] Loaded rules from config: %s", config_path)
                return rules
        except FileNotFoundError:
            logger.warning("[LOG] Rule config not found. Loading default rules.")
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
            # before optimization : SELECT * FROM (SELECT * FROM orders) AS sub WHERE sub.order_date > '2023-01-01';
            # after optimization : SELECT * FROM (SELECT * FROM orders WHERE order_date > '2023-01-01') AS sub;
            # reason : The WHERE filter now applies inside the subquery, reducing the number of rows before materialization. Predicate pushed closer to the data source.

            },
            "join_reorder": {
                "condition": "multiple JOINs",
                "action": "REORDER JOINS BY SELECTIVITY",
                "verify": "EXPLAIN COST",
                "fallback": "original_query",
                "confidence_threshold": 0.7
                # Before Optimization:SELECT * FROM large_table lt JOIN small_table st ON lt.id = st.id JOIN medium_table mt ON mt.id = st.mid;
                # After Optimization:SELECT * FROM small_table st JOIN medium_table mt ON mt.id = st.mid JOIN large_table lt ON lt.id = st.id;
                # Reason:Joins are reordered based on selectivity. Smaller, more selective tables are joined first to reduce intermediate result sizes and improve execution time.
            },
            "subquery_to_join": {
                "condition": "EXISTS subquery",
                "action": "CONVERT TO SEMI JOIN",
                "verify": "EXPLAIN COST",
                "fallback": "original_query",
                "confidence_threshold": 0.75
                # Before Optimization:SELECT customer_id FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
                # After Optimization:SELECT DISTINCT c.customer_id FROM customers c JOIN orders o ON o.customer_id = c.customer_id;
                # Reason:Rewriting EXISTS as a semi-join can be executed more efficiently. It simplifies logic and improves performance while preserving the result.
            },
            "redundant_distinct": {
                "condition": "DISTINCT with GROUP BY",
                "action": "REMOVE REDUNDANT DISTINCT",
                "verify": "LOGICAL EQUIVALENCE",
                "fallback": "original_query",
                "confidence_threshold": 0.9
                # Before Optimization:SELECT DISTINCT customer_id FROM orders GROUP BY customer_id;
                # After Optimization:SELECT customer_id FROM orders GROUP BY customer_id;
                # Reason:The GROUP BY clause already produces unique values, making the DISTINCT keyword unnecessary. Removing it avoids redundant computation.
            }
        }

    def apply_rules(self, sql: str, ast_features: Dict, similar_optimizations: List[Dict]) -> OptimizationResult:
        logger.debug("[LOG] Applying rules to SQL: %s", sql)

        try:
            ast = parse_one(sql)
            if not ast:
                logger.warning("[LOG] SQL parsing failed.")
                return None

            logger.debug("[LOG] Parsed AST: %s", ast)

            applied_rules = []
            optimized_ast = ast
            confidence_scores = []

            for rule_name, rule_config in self.rules.items():
                logger.debug("[LOG] Checking rule: %s", rule_name)

                if self._check_rule_condition(rule_config["condition"], ast, ast_features):
                    logger.debug("[LOG] Rule %s condition met", rule_name)

                    try:
                        transformed_ast = self._apply_rule_transformation(
                            optimized_ast, rule_name, rule_config, similar_optimizations
                        )
                        logger.debug("[LOG] Transformed AST using %s: %s", rule_name, transformed_ast)

                        if transformed_ast and self._verify_transformation(
                            optimized_ast, transformed_ast, rule_config["verify"]
                        ):
                            optimized_ast = transformed_ast
                            applied_rules.append(rule_name)
                            confidence_scores.append(rule_config.get("confidence_threshold", 0.7))
                            logger.debug("[LOG] Rule %s successfully applied", rule_name)

                    except Exception as e:
                        logger.warning(f"[LOG] Rule {rule_name} failed: {e}")
                        if rule_config.get("fallback") == "original_query":
                            continue

            if not applied_rules:
                logger.debug("[LOG] No rules applied.")
                return None

            optimized_sql = optimized_ast.sql(pretty=True)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            logger.debug("[LOG] Final optimized SQL: %s", optimized_sql)

            return OptimizationResult(
                optimized_sql=optimized_sql,
                optimization_type=OptimizationType.LLM_R2_REWRITE,
                confidence_score=avg_confidence,
                explanation=f"Applied LLM-R2 rules: {', '.join(applied_rules)}",
                optimization_stages=applied_rules,
                llm_reasoning=f"Rule-based optimization using dynamic patterns from {len(similar_optimizations)} similar queries"
            )

        except Exception as e:
            logger.error(f"[LOG] LLM-R2 rule application failed: {e}")
            return None

    def _check_rule_condition(self, condition: str, ast, ast_features: Dict) -> bool:
        condition_lower = condition.lower()
        logger.debug("[LOG] Checking condition: %s", condition_lower)

        if "where in subquery" in condition_lower:
            from_expr = ast.args.get("from")
            where_expr = ast.args.get("where")
            
            if not isinstance(from_expr, exp.From):
                logger.debug("[LOG] FROM clause is not present or not a valid From expression.")
                return False
            
            # Check if FROM contains exactly one subquery with alias
            subqueries = [e for e in from_expr.expressions if isinstance(e, exp.Subquery)]
            if len(subqueries) != 1:
                logger.debug("[LOG] FROM does not contain exactly one subquery.")
                return False
            
            subquery_alias = None
            # Try to find alias of the subquery
            for e in from_expr.expressions:
                if isinstance(e, exp.Subquery) and e.alias:
                    subquery_alias = e.alias
                    break
            
            if not subquery_alias:
                logger.debug("[LOG] Subquery does not have an alias.")
                return False
            
            # Check if WHERE exists and references the alias columns
            if where_expr is None:
                logger.debug("[LOG] WHERE clause does not exist.")
                return False
            
            # Simplistic check: does WHERE reference the alias name?
            # This assumes WHERE is like: sub.order_date > '2023-01-01'
            if any(
                token.this == subquery_alias.name 
                for token in where_expr.find_all(exp.Column)
                if isinstance(token.this, str)
            ):
                logger.debug("[LOG] WHERE references subquery alias columns. Condition true.")
                return True
            else:
                logger.debug("[LOG] WHERE does not reference subquery alias columns. Condition false.")
                return False



        elif "multiple joins" in condition_lower:
            joins = list(ast.find_all(exp.Join))
            logger.debug("[LOG] Number of JOIN nodes found: %d", len(joins))
            result = len(joins) >= 2
            logger.debug("[LOG] 'multiple joins' condition result: %s", result)
            return result

        elif "exists subquery" in condition_lower:
            exists_found = any(
                isinstance(sub.this, exp.Exists)
                for sub in ast.find_all(exp.Subquery)
            )
            logger.debug("[LOG] 'exists subquery' condition result: %s", exists_found)
            return exists_found

        elif "distinct with group by" in condition_lower:
            distinct_found = any(ast.find_all(exp.Distinct))
            group_found = any(ast.find_all(exp.Group))
            result = distinct_found and group_found
            logger.debug("[LOG] 'distinct with group by' condition result: %s", result)
            return result

        logger.debug("[LOG] No matching condition found for: %s", condition_lower)
        return False


    def _apply_rule_transformation(self, ast, rule_name: str, rule_config: Dict,
                                   similar_optimizations: List[Dict]):
        logger.debug("[LOG] Applying transformation for rule: %s", rule_name)

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
        if verify_method == "LOGICAL EQUIVALENCE":
            result = type(original_ast) == type(transformed_ast)
            logger.debug("[LOG] Logical equivalence check: %s", result)
            return result
        elif verify_method == "EXPLAIN COST":
            result = str(original_ast) != str(transformed_ast)
            logger.debug("[LOG] Explain cost simulation: %s", result)
            return result
        return True

    def _apply_predicate_pushdown(self, ast):
        logger.debug("[LOG] Applying manual predicate pushdown")

        if not isinstance(ast, exp.Select):
            return ast

        where_expr = ast.args.get("where")
        from_expr = ast.args.get("from")

        if not where_expr or not from_expr:
            return ast

        for subquery in from_expr.find_all(exp.Subquery):
            inner_select = subquery.this
            if isinstance(inner_select, exp.Select):
                # Push WHERE clause inside the subquery
                inner_where = inner_select.args.get("where")

                if inner_where:
                    # Combine outer and inner WHERE with AND
                    new_where = exp.and_(inner_where, where_expr)
                else:
                    new_where = where_expr

                inner_select.set("where", new_where)
                # Remove the outer WHERE
                ast.set("where", None)
                logger.debug("[LOG] Predicate pushed down successfully")

        return ast


    def _apply_join_reorder(self, ast, similar_optimizations: List[Dict]):
        logger.debug("[LOG] Before join reorder:\n%s", ast.sql())
        try:
            optimized_ast = sqlglot_optimize(ast, rules=["join_reorder"])
            logger.debug("[LOG] After join reorder:\n%s", optimized_ast.sql())
            return optimized_ast
        except Exception as e:
            logger.error("[LOG] join_reorder failed: %s", e)
            return ast  # fallback


    def _apply_subquery_to_join(self, ast):
        logger.debug("[LOG] Running sqlglot rule: unnest_subqueries")
        return sqlglot_optimize(ast, rules=["unnest_subqueries"])

    def _apply_remove_redundant_distinct(self, ast):
        logger.debug("[LOG] Manually removing redundant DISTINCT in SELECT + GROUP BY")
        for select in ast.find_all(exp.Select):
            if select.distinct and any(select.find_all(exp.Group)):
                select.set("distinct", None)
        return ast