"""
Тесты параметризации.
"""

import numpy as np

from src.parameterizer import FeatureMatrix, Parameterizer


class TestParameterizer:
    """Тесты Parameterizer."""

    def test_product_to_features_basic(self):
        """Тест базового преобразования."""
        param = Parameterizer()
        features = param.product_to_features(
            {
                "id": "test_001",
                "name": "Сплит-система Test Model",
                "characteristics": {
                    "Эффективен для помещ. площадью до": 35,
                    "Номинальная производительность охлаждения": 3.5,
                    "Уровень шума внутр. блока": 26,
                },
            }
        )

        assert features.product_id == "test_001"
        assert isinstance(features.numeric_values, np.ndarray)

    def test_product_to_features_nan_handling(self):
        """Тест обработки пропусков."""
        param = Parameterizer()
        features = param.product_to_features(
            {
                "id": "test_002",
                "name": "Minimal Product",
                "characteristics": {},
            }
        )

        assert len(features.missing_features) > 0

    def test_build_catalog_matrix(self):
        """Тест построения матрицы."""
        param = Parameterizer()
        catalog = [
            {
                "id": f"prod_{i}",
                "name": f"Test Product {i}",
                "characteristics": {
                    "Эффективен для помещ. площадью до": 25 + i * 5,
                    "Номинальная производительность охлаждения": 2.5 + i * 0.3,
                },
            }
            for i in range(10)
        ]

        matrix = param.build_catalog_matrix(catalog)

        assert isinstance(matrix, FeatureMatrix)
        assert len(matrix.product_ids) == 10
        assert matrix.numeric_matrix.shape[0] == 10

    def test_build_empty_catalog(self):
        """Тест пустого каталога."""
        param = Parameterizer()
        matrix = param.build_catalog_matrix([])

        assert matrix is not None
        assert len(matrix.product_ids) == 0
