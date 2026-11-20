"""
Тесты текстового мэтчинга.
"""

import numpy as np

from src.text_matcher import EnhancedTextMatcher, TextMatcher


class TestTextMatcher:
    """Тесты TextMatcher."""

    def test_fit_creates_vectorizer(self):
        """Тест создания векторизатора."""
        catalog = [
            "сплит система hitair ham 12",
            "daichi bravo ksgb 35 hfan",
            "ballu bsw 09hn8 white сплит",
            "кондиционер lg dual inverter",
        ]

        matcher = TextMatcher(max_features=100)
        matcher.fit(catalog)

        assert matcher.vectorizer is not None

    def test_get_similarities_shape(self):
        """Тест формы оценок."""
        catalog = [
            "сплит система hitair ham 12",
            "daichi bravo ksgb 35 hfan",
            "ballu bsw 09hn8 white сплит",
            "кондиционер lg dual inverter",
        ]

        matcher = TextMatcher(max_features=100)
        matcher.fit(catalog)
        scores = matcher.get_similarities("daichi bravo")

        assert len(scores) == 4
        assert all(0 <= score <= 1 for score in scores)

    def test_get_top_k(self):
        """Тест top-K."""
        catalog = [
            "сплит система hitair ham 12",
            "daichi bravo ksgb 35 hfan",
            "ballu bsw 09hn8 white сплит",
        ]

        matcher = TextMatcher(max_features=100)
        matcher.fit(catalog)
        indices, scores = matcher.get_top_k("ballu", k=2)

        assert len(indices) <= 2
        assert len(scores) == len(indices)

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_get_query_terms(self):
        """Тест терминов запроса."""
        matcher = TextMatcher(max_features=100)
        matcher.fit(["daichi bravo ksgb 35"])
        terms = matcher.get_query_terms("daichi bravo")

        assert isinstance(terms, dict)
        assert all(isinstance(term, str) for term in terms.keys())


class TestEnhancedTextMatcher:
    """Тесты EnhancedTextMatcher."""

    def test_fit_initializes_components(self):
        """Тест инициализации."""
        catalog = [
            {"id": "p1", "name": "Daichi Bravo KSGB35", "normalized_name": "daichi bravo ksgb 35"},
            {"id": "p2", "name": "Ballu BSW-09HN8", "normalized_name": "ballu bsw 09 hn8"},
        ]

        matcher = EnhancedTextMatcher()
        matcher.fit(catalog)

        assert matcher.tfidf_matcher is not None
        assert len(matcher.catalog) == 2

    def test_get_score_returns_array(self):
        """Тест возврата массива."""
        catalog = [
            {"id": "p1", "name": "Daichi Bravo KSGB35", "normalized_name": "daichi bravo ksgb 35"},
            {"id": "p2", "name": "Ballu BSW-09HN8", "normalized_name": "ballu bsw 09 hn8"},
        ]

        matcher = EnhancedTextMatcher()
        matcher.fit(catalog)
        scores = matcher.get_score("daichi")

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
