"""
Тесты предобработки текста.
"""

from src.preprocessor import (
    CharacteristicsNormalizer,
    ParsedProductName,
    TextPreprocessor,
)


class TestTextPreprocessor:
    """Тесты TextPreprocessor."""

    def test_basic_cleaning_lowercase(self):
        """Тест нижнего регистра."""
        processor = TextPreprocessor(use_lemmatization=False)
        result = processor.process("Сплит-система HITAIR")

        assert "hitair" in result.normalized
        assert result.original == "Сплит-система HITAIR"

    def test_empty_input(self):
        """Тест пустого входа."""
        processor = TextPreprocessor(use_lemmatization=False)
        result = processor.process("")

        assert result.original == ""
        assert result.normalized == ""

    def test_extract_brand_simple(self):
        """Тест извлечения бренда."""
        processor = TextPreprocessor(use_lemmatization=False)
        result = processor.process("Сплит-система Daichi Bravo KSGB35")

        assert result.brand == "daichi"

    def test_extract_model_number(self):
        """Тест извлечения модели."""
        processor = TextPreprocessor(use_lemmatization=False)
        result = processor.process("Daichi Bravo KSGB35HFAN1")

        assert result.model_number is not None
        assert "KSGB" in str(result.model_number).upper()

    def test_tokenize_simple(self):
        """Тест токенизации."""
        processor = TextPreprocessor(use_lemmatization=False)
        tokens = processor.tokenize("daichi bravo ksgb35 hfan1")

        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

    def test_batch_process(self):
        """Тест пакетной обработки."""
        results = TextPreprocessor.batch_process(["Daichi Bravo KSGB35", "Ballu BSW-09HN8"])

        assert len(results) == 2
        assert all(isinstance(result, ParsedProductName) for result in results)


class TestCharacteristicsNormalizer:
    """Тесты нормализатора характеристик."""

    def test_normalize_decimal_comma(self):
        """Тест десятичной запятой."""
        assert CharacteristicsNormalizer.normalize_numeric("3,55") == 3.55

    def test_normalize_with_unit(self):
        """Тест числа с единицей."""
        result = CharacteristicsNormalizer.normalize_numeric("35 квт")

        assert result is not None
        assert abs(result - 35) < 0.1

    def test_normalize_negative(self):
        """Тест отрицательного числа."""
        assert CharacteristicsNormalizer.normalize_numeric("-7") == -7

    def test_normalize_invalid(self):
        """Тест некорректного ввода."""
        assert CharacteristicsNormalizer.normalize_numeric("не число") is None

    def test_normalize_bool_yes(self):
        """Тест положительного булева значения."""
        assert CharacteristicsNormalizer.normalize_boolean("Да") is True

    def test_normalize_bool_no(self):
        """Тест отрицательного булева значения."""
        assert CharacteristicsNormalizer.normalize_boolean("Нет") is False

    def test_parse_temp_range_simple(self):
        """Тест диапазона температур."""
        min_t, max_t = CharacteristicsNormalizer.parse_temperature_range("-7 +43")

        assert abs(min_t - (-7)) < 0.1
        assert abs(max_t - 43) < 0.1

    def test_parse_dimension_standard(self):
        """Тест стандартной размерности."""
        result = CharacteristicsNormalizer.parse_dimension("0.462 × 0.68 × 0.25")

        assert abs(result["height"] - 0.462) < 0.001
        assert abs(result["width"] - 0.68) < 0.001
        assert abs(result["depth"] - 0.25) < 0.001
