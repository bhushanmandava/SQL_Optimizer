import logging
from typing import Dict
from difflib import SequenceMatcher

import sqlglot
from sqlglot import parse_one, exp

from Schema import ASTShape,logger

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
                'aggregates': len([
                    f for f in ast.find_all(exp.Func)
                    if hasattr(f, 'is_aggregate') and f.is_aggregate
                ]),
                'union_count': len(list(ast.find_all(exp.Union))),
                'where_conditions': len(list(ast.find_all(exp.Where))),
                'having_conditions': len(list(ast.find_all(exp.Having))),
                'order_by_columns': len(list(ast.find_all(exp.Order))),
                'distinct_usage': len(list(ast.find_all(exp.Distinct))),
                'case_statements': len(list(ast.find_all(exp.Case))),
                'nested_depth': self._calculate_nesting_depth(ast),
                'complexity_score': self._calculate_complexity_score(ast),
                'table_references': [
                    table.this.this for table in ast.find_all(exp.Table)
                    if isinstance(table.this, exp.Identifier)
                ]
            }

            return features
        except Exception as e:
            logger.exception(f"Feature extraction failed: {e}")
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
        if isinstance(node, exp.Identifier):
            node.set("this", node.this.lower())
        elif isinstance(node, exp.Table):
            if isinstance(node.this, exp.Identifier):
                node.this.set("this", node.this.this.lower())
        elif isinstance(node, exp.Column):
            if isinstance(node.this, exp.Identifier):
                node.this.set("this", node.this.this.lower())
        return node


    
    def _calculate_nesting_depth(self, ast) -> int:
        def depth(node):
            if not hasattr(node, 'args') or not node.args:
                return 1
            max_child_depth = 0
            for child in node.args.values():
                if isinstance(child, list):
                    max_child_depth = max(max_child_depth, max(depth(c) for c in child if isinstance(c, exp.Expression)))
                elif isinstance(child, exp.Expression):
                    max_child_depth = max(max_child_depth, depth(child))
            return 1 + max_child_depth
        return depth(ast)

    
    def _calculate_complexity_score(self, ast) -> float:
        """Calculate query complexity score based on multiple factors"""
        features = {
            'tables': len(list(ast.find_all(exp.Table))),
            'joins': len(list(ast.find_all(exp.Join))),
            'subqueries': len(list(ast.find_all(exp.Subquery))),
            'aggregates': len([f for f in ast.find_all(exp.Func) if hasattr(f, 'is_aggregate') and f.is_aggregate]),
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
        if type(ast1) != type(ast2):
            return 0.0

        if not hasattr(ast1, 'args') or not hasattr(ast2, 'args'):
            return 1.0

        keys1, keys2 = set(ast1.args.keys()), set(ast2.args.keys())
        common_keys = keys1 & keys2
        total_keys = keys1 | keys2

        if not total_keys:
            return 1.0

        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = ast1.args[key], ast2.args[key]
            # Handle list children
            if isinstance(val1, list) and isinstance(val2, list):
                matches = 0
                for c1, c2 in zip(val1, val2):
                    if isinstance(c1, exp.Expression) and isinstance(c2, exp.Expression):
                        matches += self._compute_structural_similarity(c1, c2)
                    else:
                        matches += float(c1 == c2)
                similarity_sum += matches / max(len(val1), len(val2))
            elif isinstance(val1, exp.Expression) and isinstance(val2, exp.Expression):
                similarity_sum += self._compute_structural_similarity(val1, val2)
            else:
                similarity_sum += float(val1 == val2)

        return similarity_sum / len(total_keys)

    
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
