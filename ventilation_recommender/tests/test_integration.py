"""
Интеграционные тесты.
"""

import time

import numpy as np
import pytest

from src.hybrid_model import AnalogRecommender, normalize_distance_to_similarity


@pytest.fixture
def sample_catalog():
    """Тестовый каталог."""
    return [
        {
            "id": "prod_001",
            "name": "Сплит-система HITAIR HAM-12H/N1_V2 комплект",
            "code": "НС-1762906",
            "characteristics": {
                "Эффективен для помещ. площадью до": 35,
                "Номинальная производительность охлаждения": 3.55,
                "Уровень шума внутр. блока": 26,
                "Бренд": "HITAIR",
            },
        },
        {
            "id": "prod_002",
            "name": "Daichi Bravo KSGB35HFAN1/KSRB35HFAN1",
            "code": "НС-1234567",
            "characteristics": {
                "Эффективен для помещ. площадью до": 35,
                "Номинальная производительность охлаждения": 3.5,
                "Уровень шума внутр. блока": 31,
                "Бренд": "Daichi",
            },
        },
        {
            "id": "prod_003",
            "name": "Ballu BSW-09HN8_20Y комплект сплит-система",
            "code": "НС-7654321",
            "characteristics": {
                "Эффективен для помещ. площадью до": 26,
                "Номинальная производительность охлаждения": 2.6,
                "Уровень шума внутр. блока": 24,
                "Бренд": "Ballu",
            },
        },
    ]


@pytest.mark.asyncio
async def test_initialize_success(sample_catalog):
    """Тест инициализации."""
    rec = AnalogRecommender()

    assert not rec._initialized

    await rec.initialize(sample_catalog)

    assert rec._initialized
    assert rec.catalog_size == 3


@pytest.mark.asyncio
async def test_find_analogs_by_name(sample_catalog):
    """Тест поиска по названию."""
    rec = AnalogRecommender(text_weight=0.4, param_weight=0.6)
    await rec.initialize(sample_catalog)

    results = await rec.find_analogs("Daichi Bravo KSGB35HFAN1", top_k=5, min_score=0.05)

    assert len(results.results) > 0


@pytest.mark.asyncio
async def test_find_analogs_by_id(sample_catalog):
    """Тест поиска по ID."""
    rec = AnalogRecommender(text_weight=0.4, param_weight=0.6)
    await rec.initialize(sample_catalog)

    results = await rec.find_analogs("prod_002", top_k=5, min_score=0.05)

    assert isinstance(results.total_catalog_size, int)


@pytest.mark.asyncio
async def test_find_analogs_top_k_limit(sample_catalog):
    """Тест ограничения top-K."""
    rec = AnalogRecommender(text_weight=0.4, param_weight=0.6)
    await rec.initialize(sample_catalog)

    results = await rec.find_analogs("сплит система", top_k=2, min_score=0.0)

    assert len(results.results) <= 2


@pytest.mark.asyncio
async def test_find_analogs_min_score_threshold(sample_catalog):
    """Тест порога."""
    rec = AnalogRecommender(text_weight=0.4, param_weight=0.6)
    await rec.initialize(sample_catalog)

    results = await rec.find_analogs("что-то очень редкое xyz", top_k=10, min_score=0.5)

    for result in results.results:
        assert result.score >= 0.5


@pytest.mark.asyncio
async def test_find_analogs_timing(sample_catalog):
    """Тест времени выполнения."""
    rec = AnalogRecommender(text_weight=0.4, param_weight=0.6)
    await rec.initialize(sample_catalog)

    start = time.time()
    await rec.find_analogs("Daichi Bravo", top_k=10, min_score=0.05)
    elapsed_ms = (time.time() - start) * 1000

    assert elapsed_ms < 500


@pytest.mark.asyncio
async def test_get_stats(sample_catalog):
    """Тест статистики."""
    rec = AnalogRecommender(text_weight=0.4, param_weight=0.6)
    await rec.initialize(sample_catalog)

    stats = rec.get_stats()

    assert stats["initialized"] is True
    assert stats["catalog_size"] > 0
    assert "weights" in stats


def test_normalize_distance_basic():
    """Тест нормализации расстояния."""
    assert abs(normalize_distance_to_similarity(0, scale_factor=2.0) - 1.0) < 0.001
    assert normalize_distance_to_similarity(10, scale_factor=2.0) < 0.01


def test_normalize_distance_array():
    """Тест нормализации массива."""
    distances = np.array([0, 1, 5, 10], dtype=float)
    similarities = normalize_distance_to_similarity(distances, scale_factor=2.0)

    assert len(similarities) == 4
    assert all(0 <= score <= 1 for score in similarities)
    assert similarities[0] >= max(similarities)
