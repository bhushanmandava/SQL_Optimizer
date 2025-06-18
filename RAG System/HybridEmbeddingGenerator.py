import torch
import numpy as np
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
from Schema import logger

class HybridEmbeddingGenerator:
    """Hybrid embedding system combining CodeBERT and graph features"""

    def __init__(self, config: Dict):
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Initialize CodeBERT
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)

        # Embedding weights - adjust as needed
        self.codebert_weight = config.get('codebert_weight', 0.6)  # Adjusted weights since T5 is removed
        # self.t5_weight = config.get('t5_weight', 0.3)  # Commented out
        self.graph_weight = config.get('graph_weight', 0.4)

        # Dimension consistency
        self.target_dim = 768

    def generate_embedding(self, sql: str, ast_features: Dict = None) -> List[float]:
        """Generate hybrid embedding combining CodeBERT and graph features"""
        try:
            codebert_emb = self._get_codebert_embedding(sql)
            # t5_emb = self._get_t5_embedding(sql)  # Commented out
            graph_emb = self._get_graph_embedding(ast_features or {})

            hybrid_emb = (
                self.codebert_weight * codebert_emb +
                # self.t5_weight * t5_emb +  # Commented out
                self.graph_weight * graph_emb
            )

            hybrid_emb = hybrid_emb / np.linalg.norm(hybrid_emb)
            return hybrid_emb.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * self.target_dim

    def _get_codebert_embedding(self, sql: str) -> np.ndarray:
        try:
            inputs = self.codebert_tokenizer(
                sql, return_tensors="pt", max_length=512, truncation=True, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.codebert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"CodeBERT embedding failed: {e}")
            return np.zeros(self.target_dim)

    # Commented out T5 embedding method
    # def _get_t5_embedding(self, sql: str) -> np.ndarray:
    #     try:
    #         input_text = f"translate to SQL: {sql}"
    #         inputs = self.t5_tokenizer(
    #             input_text, return_tensors="pt", max_length=512, truncation=True, padding=True
    #         ).to(self.device)
    #
    #         with torch.no_grad():
    #             encoder_outputs = self.t5_model.encoder(**inputs)
    #             embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    #
    #         return embeddings.squeeze().cpu().numpy()
    #
    #     except Exception as e:
    #         logger.error(f"T5 embedding failed: {e}")
    #         return np.zeros(self.target_dim)

    def _get_graph_embedding(self, ast_features: Dict) -> np.ndarray:
        try:
            feature_vector = np.array([
                ast_features.get('table_count', 0),
                ast_features.get('join_count', 0),
                ast_features.get('subquery_count', 0),
                ast_features.get('aggregate_count', 0),
                ast_features.get('window_function_count', 0),
                ast_features.get('cte_count', 0),
                ast_features.get('union_count', 0),
                ast_features.get('complexity_score', 0) / 100.0,
                ast_features.get('nested_depth', 0) / 10.0,
                ast_features.get('where_conditions', 0),
                ast_features.get('having_conditions', 0),
                ast_features.get('order_by_columns', 0),
                ast_features.get('distinct_usage', 0),
                ast_features.get('case_statements', 0)
            ], dtype=np.float32)

            if len(feature_vector) < self.target_dim:
                expansion_factor = self.target_dim // len(feature_vector)
                remainder = self.target_dim % len(feature_vector)
                expanded = np.tile(feature_vector, expansion_factor)
                if remainder > 0:
                    expanded = np.concatenate([expanded, feature_vector[:remainder]])
                noise = np.random.normal(0, 0.01, self.target_dim)
                return expanded + noise
            else:
                return feature_vector[:self.target_dim]

        except Exception as e:
            logger.error(f"Graph embedding failed: {e}")
            return np.zeros(self.target_dim)
