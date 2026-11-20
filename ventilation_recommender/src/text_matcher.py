"""
Текстовый мэтчинг на основе TF-IDF.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import TFIDF_CONFIG


class TextMatcher:
    """TF-IDF текстовый мэтчер."""

    def __init__(
        self,
        max_features: int = TFIDF_CONFIG["max_features"],
        ngram_range: Tuple[int, int] = TFIDF_CONFIG["ngram_range"],
        min_df: int = TFIDF_CONFIG["min_df"],
        max_df: float = TFIDF_CONFIG["max_df"],
        sublinear_tf: bool = TFIDF_CONFIG["sublinear_tf"],
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.catalog_matrix = None

    def fit(self, texts: List[str]) -> "TextMatcher":
        """Обучает TF-IDF модель."""
        clean_texts = [text if isinstance(text, str) else "" for text in texts]
        if not clean_texts:
            self.catalog_matrix = None
            return self
        effective_min_df = 1 if len(clean_texts) < 3 else self.min_df
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=effective_min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            analyzer="word",
            token_pattern=r"\b\w+\b",
        )

        try:
            self.catalog_matrix = self.vectorizer.fit_transform(clean_texts)
        except ValueError:
            self.catalog_matrix = None
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Трансформирует тексты в TF-IDF векторы."""
        if self.vectorizer is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        return self.vectorizer.transform(texts)

    def get_similarities(self, query: str) -> np.ndarray:
        """Вычисляет косинусное сходство запроса с каталогом."""
        if self.catalog_matrix is None:
            return np.array([], dtype=float)
        query_vector = self.vectorizer.transform([query or ""])
        return cosine_similarity(query_vector, self.catalog_matrix)[0]

    def get_top_k(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает top-K индексы и оценки."""
        similarities = self.get_similarities(query)
        if len(similarities) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        sorted_indices = np.argsort(similarities)[::-1]
        valid_indices = [idx for idx in sorted_indices if similarities[idx] >= min_score][:k]
        return np.array(valid_indices, dtype=int), np.array([similarities[idx] for idx in valid_indices], dtype=float)

    def get_vocabulary(self) -> List[str]:
        """Возвращает словарь признаков."""
        if self.vectorizer is None:
            return []
        return list(self.vectorizer.vocabulary_.keys())

    def get_query_terms(self, text: str) -> Dict[str, float]:
        """Возвращает TF-IDF веса терминов запроса."""
        if self.vectorizer is None:
            return {}
        query_vector = self.vectorizer.transform([text or ""])
        feature_names = self.vectorizer.get_feature_names_out()
        terms = {
            str(feature_names[idx]): float(score)
            for idx, score in zip(query_vector.indices, query_vector.data)
        }
        return dict(sorted(terms.items(), key=lambda item: -item[1]))


class EnhancedTextMatcher:
    """Расширенный текстовый мэтчер."""

    def __init__(self):
        self.tfidf_matcher = TextMatcher()
        self.catalog: List[Dict] = []

    def fit(self, catalog: List[Dict], text_field: str = "normalized_name") -> "EnhancedTextMatcher":
        """Обучает мэтчер на каталоге."""
        self.catalog = catalog
        texts = [item.get(text_field, "") for item in catalog]
        self.tfidf_matcher.fit(texts)
        return self

    def get_score(self, query: str) -> np.ndarray:
        """Возвращает текстовые оценки для каталога."""
        if not self.catalog:
            return np.array([], dtype=float)
        scores = self.tfidf_matcher.get_similarities(query)
        if len(scores) == 0:
            return np.zeros(len(self.catalog), dtype=float)
        return scores

    def explain_score(self, query: str, product_idx: int) -> Dict:
        """Объясняет оценку для товара."""
        if not self.catalog or product_idx >= len(self.catalog):
            return {}
        product = self.catalog[product_idx]
        query_terms = self.tfidf_matcher.get_query_terms(query)
        catalog_name = product.get("normalized_name", "")
        catalog_terms = self.tfidf_matcher.get_query_terms(catalog_name)
        common_terms = set(query_terms.keys()) & set(catalog_terms.keys())
        scores = self.tfidf_matcher.get_similarities(query)
        tfidf_score = float(scores[product_idx]) if len(scores) > product_idx else 0.0
        return {
            "product_name": product.get("name"),
            "tfidf_score": tfidf_score,
            "query_terms": query_terms,
            "catalog_terms": catalog_terms,
            "common_terms": list(common_terms),
            "brand_match": bool(product.get("brand") and product["brand"].lower() in (query or "").lower()),
        }
