"""
Параметрический мэтчинг товаров.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .config import HYBRID_WEIGHTS


@dataclass
class MatchResult:
    """Результат сравнения товаров по параметрам."""
    product_idx: int
    overall_score: float
    numeric_distance: Optional[float]
    categorical_similarity: Optional[float]
    feature_distances: Dict[str, float]


@dataclass
class MatchResults:
    """Результаты параметрического мэтчинга."""
    results: List[MatchResult]
    best_match_idx: int
    best_score: float

    def get_top_k(self, k: int) -> List[MatchResult]:
        """Возвращает top-K результатов."""
        sorted_results = sorted(self.results, key=lambda item: -item.overall_score)
        return sorted_results[:k]


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Евклидово расстояние."""
    if len(vec1) != len(vec2):
        raise ValueError("Векторы должны быть одинаковой длины")
    return float(np.sqrt(np.sum((vec1 - vec2) ** 2)))


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Манхэттенское расстояние."""
    if len(vec1) != len(vec2):
        raise ValueError("Векторы должны быть одинаковой длины")
    return float(np.sum(np.abs(vec1 - vec2)))


def cosine_similarity_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Косинусное расстояние."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(1 - similarity)


def mahalanobis_distance(point: np.ndarray, data: np.ndarray, cov_inv: Optional[np.ndarray] = None) -> float:
    """Расстояние Махаланобиса."""
    if cov_inv is None:
        cov = np.cov(data, rowvar=False)
        try:
            cov_inv = np.linalg.inv(cov + np.eye(len(cov)) * 1e-6)
        except np.linalg.LinAlgError:
            return euclidean_distance(point, data.mean(axis=0))
    diff = point - data.mean(axis=0)
    return float(np.sqrt(diff @ cov_inv @ diff.T))


def weighted_minkowski(
    vec1: np.ndarray,
    vec2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    p: float = 2.0,
) -> float:
    """Взвешенное расстояние Минковского."""
    if len(vec1) != len(vec2):
        raise ValueError("Векторы должны быть одинаковой длины")
    if weights is None:
        weights = np.ones(len(vec1))
    diff = np.abs(vec1 - vec2)
    if p == float("inf"):
        return float(np.max(weights * diff))
    return float(np.sum((weights * diff ** p)) ** (1 / p))


def normalize_distance_to_similarity(distance: float | np.ndarray, scale_factor: float = 2.0) -> float | np.ndarray:
    """Преобразует расстояние в сходство."""
    return np.exp(-np.asarray(distance, dtype=float) * scale_factor)


class ParamMatcher:
    """Параметрический мэтчер."""

    def __init__(self, distance_metric: str = "euclidean", use_mahalanobis: bool = False):
        self.distance_metric = distance_metric
        self.use_mahalanobis = use_mahalanobis
        self.catalog_matrix: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.n_features = 0

    def set_catalog(self, feature_matrix: np.ndarray) -> "ParamMatcher":
        """Устанавливает каталог признаков."""
        self.catalog_matrix = np.asarray(feature_matrix, dtype=float)
        if self.catalog_matrix.ndim == 1:
            self.catalog_matrix = self.catalog_matrix.reshape(1, -1)
        self.n_features = self.catalog_matrix.shape[1] if self.catalog_matrix.ndim > 1 else 0
        if self.use_mahalanobis and self.catalog_matrix.size > 0:
            try:
                cov = np.cov(self.catalog_matrix, rowvar=False)
                cov_reg = cov + np.eye(self.n_features) * 1e-6
                self.cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                self.use_mahalanobis = False
        return self

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет расстояние между векторами."""
        if self.distance_metric == "euclidean":
            return euclidean_distance(vec1, vec2)
        if self.distance_metric == "manhattan":
            return manhattan_distance(vec1, vec2)
        if self.distance_metric == "cosine":
            return cosine_similarity_distance(vec1, vec2)
        return euclidean_distance(vec1, vec2)

    def get_distances(self, query_vector: np.ndarray) -> np.ndarray:
        """Возвращает расстояния до всех товаров."""
        if self.catalog_matrix is None:
            raise RuntimeError("Каталог не установлен. Вызовите set_catalog()")
        if len(self.catalog_matrix) == 0:
            return np.array([], dtype=float)
        query_vector = np.nan_to_num(np.asarray(query_vector, dtype=float).flatten(), nan=0.0)
        if len(query_vector) != self.n_features:
            if len(query_vector) < self.n_features:
                query_vector = np.pad(query_vector, (0, self.n_features - len(query_vector)), mode="constant")
            else:
                query_vector = query_vector[: self.n_features]
        distances = np.array(
            [self.compute_distance(query_vector, catalog_vec) for catalog_vec in self.catalog_matrix],
            dtype=float,
        )

        return np.nan_to_num(distances, nan=0.0)

    def get_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """Возвращает параметрические оценки близости."""
        distances = self.get_distances(query_vector)
        if len(distances) == 0:
            return np.array([], dtype=float)
        positive_distances = distances[distances > 0]
        mean_dist = float(np.mean(positive_distances)) if len(positive_distances) > 0 else 1.0
        scale_factor = HYBRID_WEIGHTS.numeric_weight * max(self.n_features, 1) / max(mean_dist, 0.01)
        return np.asarray(normalize_distance_to_similarity(distances, scale_factor=scale_factor), dtype=float)

    def get_top_k(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает top-K параметров."""
        similarities = self.get_similarities(query_vector)
        if len(similarities) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        sorted_indices = np.argsort(similarities)[::-1][:k]
        return sorted_indices.astype(int), similarities[sorted_indices]

    def get_feature_contributions(self, query_vector: np.ndarray, top_n_features: int = 5) -> List[Dict]:
        """Анализирует вклад признаков."""
        if self.catalog_matrix is None or len(self.catalog_matrix) == 0:
            return []
        query = np.nan_to_num(np.asarray(query_vector, dtype=float).flatten(), nan=0.0)
        contributions = []
        for j in range(self.n_features):
            feature_values = self.catalog_matrix[:, j]
            mean_val = float(np.mean(feature_values))
            std_val = float(np.std(feature_values))
            diff = abs(query[j] - mean_val) if j < len(query) else 0.0
            contributions.append(
                {
                    "feature_idx": j,
                    "query_value": float(query[j]) if j < len(query) else 0.0,
                    "catalog_mean": mean_val,
                    "catalog_std": std_val,
                    "contribution": float(diff),
                    "normalized_contribution": float(diff / (std_val + 1e-6)),
                }
            )
        contributions = sorted(contributions, key=lambda item: -item["contribution"])
        return contributions[:top_n_features]


class HybridParamMatcher:
    """Гибридный параметрический мэтчер."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "numeric": HYBRID_WEIGHTS.numeric_weight,
            "categorical": HYBRID_WEIGHTS.categorical_weight,
        }
        self.numeric_matcher = ParamMatcher(distance_metric="euclidean")
        self.categorical_matrix: Optional[np.ndarray] = None

    def fit(self, numeric_matrix: np.ndarray, categorical_matrix: np.ndarray) -> "HybridParamMatcher":
        """Обучает гибридный параметрический мэтчер."""
        if numeric_matrix is not None and len(numeric_matrix) > 0:
            self.numeric_matcher.set_catalog(numeric_matrix)
        self.categorical_matrix = categorical_matrix
        return self

    def compute_categorical_similarity(self, query_cat: np.ndarray, catalog_idx: int) -> float:
        """Вычисляет сходство по категориальным признакам."""
        if self.categorical_matrix is None or len(self.categorical_matrix) == 0:
            return 1.0
        cat_vector = self.categorical_matrix[catalog_idx]
        similarities = []
        for q_val, c_val in zip(query_cat, cat_vector):
            if q_val < 0 or c_val < 0:
                continue
            similarities.append(1.0 if abs(q_val - c_val) < 1e-6 else 0.0)
        return float(np.mean(similarities)) if similarities else 0.5

    def get_all_scores(
        self,
        query_numeric: np.ndarray,
        query_categorical: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Возвращает комбинированные параметрические оценки."""
        if self.numeric_matcher.catalog_matrix is None or len(self.numeric_matcher.catalog_matrix) == 0:
            return np.array([], dtype=float)
        numeric_scores = np.asarray(
            normalize_distance_to_similarity(self.numeric_matcher.get_distances(query_numeric), scale_factor=2.0),
            dtype=float,
        )
        if query_categorical is None or self.categorical_matrix is None:
            return numeric_scores
        n = len(numeric_scores)
        categorical_scores = np.array(
            [self.compute_categorical_similarity(query_categorical, i) for i in range(n)],
            dtype=float,
        )
        w_num = self.weights.get("numeric", 0.7)
        w_cat = self.weights.get("categorical", 0.3)
        return w_num * numeric_scores + w_cat * categorical_scores

    def get_top_k(
        self,
        query_numeric: np.ndarray,
        query_categorical: Optional[np.ndarray] = None,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает top-K гибридных параметров."""
        scores = self.get_all_scores(query_numeric, query_categorical)
        if len(scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        sorted_indices = np.argsort(scores)[::-1][:k]
        return sorted_indices.astype(int), scores[sorted_indices]
