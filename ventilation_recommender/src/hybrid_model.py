"""
Гибридная модель подбора аналогов.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from .config import HYBRID_WEIGHTS, MIN_SCORE_THRESHOLD, TOP_K_DEFAULT
from .parameterizer import Parameterizer
from .param_matcher import ParamMatcher, normalize_distance_to_similarity
from .preprocessor import TextPreprocessor
from .text_matcher import EnhancedTextMatcher


@dataclass
class AnalogResult:
    """Один найденный аналог."""
    product_id: str
    name: str
    code: Optional[str]
    score: float
    text_score: float
    param_score: float
    match_details: Dict = field(default_factory=dict)


@dataclass
class AnalogResults:
    """Результаты подбора аналогов."""
    query: str
    timestamp: str
    results: List[AnalogResult]
    total_catalog_size: int
    execution_time_ms: float

    @property
    def is_empty(self) -> bool:
        """Пустой ли результат."""
        return len(self.results) == 0


class AnalogRecommender:
    """Гибридный рекомендатель аналогов."""

    def __init__(
        self,
        text_weight: float = HYBRID_WEIGHTS.text_similarity,
        param_weight: float = HYBRID_WEIGHTS.param_distance,
    ):
        total_weight = max(text_weight + param_weight, 1e-6)
        self.text_weight = text_weight / total_weight
        self.param_weight = param_weight / total_weight
        self._initialized = False
        self._catalog: List[Dict] = []
        self._processed_catalog: List[Dict] = []
        self._id_to_index: Dict[str, int] = {}
        self.preprocessor: Optional[TextPreprocessor] = None
        self.parameterizer: Optional[Parameterizer] = None
        self.text_matcher: Optional[EnhancedTextMatcher] = None
        self.param_matcher: Optional[ParamMatcher] = None

    async def initialize(self, catalog_data: List[Dict]) -> "AnalogRecommender":
        """Инициализирует рекомендатель каталогом."""
        start_time = time.time()
        self._catalog = catalog_data
        self.preprocessor = TextPreprocessor(use_lemmatization=True)
        processed_catalog: List[Dict] = []
        self._id_to_index = {}
        for product in catalog_data:
            name_result = self.preprocessor.process(product.get("name", product.get("Name", "")))
            product_id = str(product.get("id", product.get("_id", "")))
            if not product_id and isinstance(product.get("_id"), dict):
                product_id = str(product["_id"].get("$uuid", ""))
            processed_product = {
                "id": product_id,
                "name": str(product.get("name", product.get("Name", ""))),
                "code": str(product.get("code", product.get("Code", ""))),
                "normalized_name": name_result.normalized,
                "brand": name_result.brand,
                "model_number": name_result.model_number,
            }
            if isinstance(product.get("characteristics"), dict):
                processed_product["characteristics"] = product["characteristics"]
            elif isinstance(product.get("Characteristics"), dict):
                processed_product["characteristics"] = product["Characteristics"]
            elif isinstance(product.get("Characteristics"), list):
                processed_product["characteristics"] = self._parse_characteristics(product["Characteristics"])
            if product_id:
                self._id_to_index[product_id] = len(processed_catalog)
            processed_catalog.append(processed_product)
        self._processed_catalog = processed_catalog
        self.parameterizer = Parameterizer()
        feature_matrix = self.parameterizer.build_catalog_matrix(processed_catalog)
        normalized_names = [item["normalized_name"] for item in processed_catalog]
        self.text_matcher = EnhancedTextMatcher()
        self.text_matcher.fit(processed_catalog, text_field="normalized_name")
        if feature_matrix.numeric_matrix.size > 0:
            self.param_matcher = ParamMatcher(distance_metric="euclidean")
            self.param_matcher.set_catalog(feature_matrix.numeric_matrix)
        self._initialized = True
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"Инициализация завершена за {elapsed_ms:.1f} мс. Каталог: {self.catalog_size}")
        return self

    async def find_analogs(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        min_score: float = MIN_SCORE_THRESHOLD,
        include_details: bool = False,
    ) -> AnalogResults:
        """Ищет аналоги по строке или ID."""
        start_time = time.time()
        if not self._initialized:
            raise RuntimeError("Рекомендатель не инициализирован.")
        query_product_idx = self._find_by_id(query)
        if query_product_idx is not None:
            query_product = self._processed_catalog[query_product_idx]
            text_query = query_product.get("normalized_name", query)
            query_numeric = self._get_query_numeric_by_index(query_product_idx)
            exclude_idx = query_product_idx
        else:
            parsed_query = self.preprocessor.process(query)
            text_query = parsed_query.normalized
            query_numeric = self._get_query_numeric_from_text(query, parsed_query)
            exclude_idx = None
        text_scores = self._compute_text_scores(text_query)
        param_scores = self._compute_param_scores(query_numeric)
        combined_scores = self.text_weight * text_scores + self.param_weight * param_scores
        combined_scores = np.nan_to_num(combined_scores, nan=0.0)
        sorted_indices = np.argsort(combined_scores)[::-1]
        results: List[AnalogResult] = []
        for idx in sorted_indices:
            if exclude_idx is not None and int(idx) == exclude_idx:
                continue
            score = float(combined_scores[idx])
            if score < min_score:
                break
            product = self._processed_catalog[int(idx)]
            results.append(
                AnalogResult(
                    product_id=str(product.get("id", "")),
                    name=product.get("name", ""),
                    code=product.get("code") or None,
                    score=score,
                    text_score=float(text_scores[int(idx)]),
                    param_score=float(param_scores[int(idx)]),
                    match_details=self._build_match_details(int(idx), text_query, include_details),
                )
            )
            if len(results) >= top_k:
                break
        elapsed_ms = (time.time() - start_time) * 1000
        return AnalogResults(
            query=query,
            timestamp=str(datetime.now()),
            results=results,
            total_catalog_size=self.catalog_size,
            execution_time_ms=float(elapsed_ms),
        )

    def _compute_text_scores(self, text_query: str) -> np.ndarray:
        """Вычисляет текстовые оценки."""
        if self.text_matcher is None:
            return np.zeros(self.catalog_size, dtype=float)
        scores = self.text_matcher.get_score(text_query)
        if len(scores) == 0:
            return np.zeros(self.catalog_size, dtype=float)
        return np.nan_to_num(scores, nan=0.0)

    def _compute_param_scores(self, query_numeric: np.ndarray) -> np.ndarray:
        """Вычисляет параметрические оценки."""
        if self.param_matcher is None:
            return np.full(self.catalog_size, 0.5, dtype=float)
        scores = self.param_matcher.get_similarities(query_numeric)
        if len(scores) == 0:
            return np.full(self.catalog_size, 0.5, dtype=float)
        return np.nan_to_num(scores, nan=0.5)

    def _get_query_numeric_by_index(self, idx: int) -> np.ndarray:
        """Получает числовой вектор товара из матрицы."""
        if self.parameterizer is None or self.parameterizer.feature_matrix is None:
            return np.zeros(len(self.parameterizer.numeric_feature_names if self.parameterizer else 0), dtype=float)
        matrix = self.parameterizer.feature_matrix.numeric_matrix
        if len(matrix) == 0 or idx >= len(matrix):
            return np.zeros(matrix.shape[1] if matrix.ndim > 1 else 0, dtype=float)
        return np.nan_to_num(matrix[idx], nan=0.0)

    def _get_query_numeric_from_text(self, text: str, parsed_query) -> np.ndarray:
        """Получает числовой вектор из текста."""
        if self.parameterizer is None:
            return np.array([], dtype=float)
        numeric_vector, _ = self.parameterizer.text_to_features(text, self.preprocessor)
        numeric_vector = np.nan_to_num(np.asarray(numeric_vector, dtype=float), nan=0.0)
        if self.parameterizer.feature_matrix is not None and self.parameterizer.feature_matrix.numeric_matrix.size > 0:
            if numeric_vector.size == 0 or not np.any(numeric_vector):
                numeric_vector = self.parameterizer.feature_matrix.numeric_matrix.mean(axis=0)
        return numeric_vector

    def _find_by_id(self, query: str) -> Optional[int]:
        """Ищет товар по ID."""
        if not query:
            return None
        if query in self._id_to_index:
            return self._id_to_index[query]
        for product_id, idx in self._id_to_index.items():
            if product_id.startswith(query):
                return idx
        return None

    def _parse_characteristics(self, chars_list: List[Dict]) -> Dict[str, object]:
        """Парсит список характеристик."""
        result: Dict[str, object] = {}
        for char in chars_list:
            if not isinstance(char, dict):
                continue
            name = str(char.get("Name", "")).strip()
            value = char.get("Value")
            if name and value is not None:
                result[name] = value
        return result

    def _build_match_details(self, idx: int, text_query: str, include_features: bool) -> Dict:
        """Строит детали совпадения."""
        details = {"index": idx}
        if include_features and self.text_matcher is not None:
            try:
                details["text_explanation"] = self.text_matcher.explain_score(text_query, idx)
            except Exception:
                pass
        return details

    @property
    def catalog_size(self) -> int:
        """Размер каталога."""
        return len(self._processed_catalog)

    def get_stats(self) -> Dict:
        """Возвращает статистику рекомендателя."""
        return {
            "initialized": self._initialized,
            "catalog_size": self.catalog_size,
            "weights": {
                "text": self.text_weight,
                "param": self.param_weight,
            },
            "components": {
                "preprocessor": self.preprocessor is not None,
                "parameterizer": self.parameterizer is not None,
                "text_matcher": self.text_matcher is not None,
                "param_matcher": self.param_matcher is not None,
            },
        }


def create_recommender(config: Optional[Dict] = None) -> AnalogRecommender:
    """Создаёт рекомендатель."""
    if config is None:
        config = {}
    return AnalogRecommender(
        text_weight=config.get("text_weight", HYBRID_WEIGHTS.text_similarity),
        param_weight=config.get("param_weight", HYBRID_WEIGHTS.param_distance),
    )
