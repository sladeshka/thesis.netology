"""
Предобработка текста названий товаров.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .config import REMOVE_CHARS, STOP_WORDS_RU
try:
    import pymorphy3
    MORPHY_AVAILABLE = True
except ImportError:
    pymorphy3 = None
    MORPHY_AVAILABLE = False

morph = None
if MORPHY_AVAILABLE:
    try:
        morph = pymorphy3.MorphAnalyzer()
    except Exception:
        morph = None


@dataclass
class ParsedProductName:
    """Результат разбора названия товара."""
    original: str
    normalized: str
    brand: Optional[str] = None
    model_number: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    cleaned_tokens: List[str] = field(default_factory=list)


KNOWN_BRANDS = {
    "daichi",
    "hitair",
    "ballu",
    "lg",
    "samsung",
    "panasonic",
    "mitsubishi",
    "toshiba",
    "haier",
    "electrolux",
    "zanussi",
    "lessar",
    "quattroclima",
    "kentatsu",
    "general",
    "fujitsu",
    "aux",
    "dantex",
    "centek",
    "midea",
    "abion",
    "timberk",
}

class TextPreprocessor:
    """Предобработчик названий товаров."""

    def __init__(self, use_lemmatization: bool = True, lowercase: bool = True):
        self.use_lemmatization = use_lemmatization and morph is not None
        self.lowercase = lowercase
        self._init_regex_patterns()

    def _init_regex_patterns(self) -> None:
        brand_pattern = "|".join(re.escape(brand) for brand in KNOWN_BRANDS)
        if not brand_pattern:
            brand_pattern = "unknown_brand"
        self._brand_start_pattern = re.compile(
            rf"^(сплит[\-\s]?система|кондиционер)?\s*({brand_pattern})",
            re.IGNORECASE,
        )
        self._model_pattern = re.compile(
            r"\b([A-Z]{2,}[-\s]?[A-Z]{0,3}[-\s]?\d{1,4}[A-Z]*)\b",
            re.IGNORECASE,
        )
        self._btu_pattern = re.compile(
            r"\b(\d{2,5})\s*(btu|тыс\.?\s*бти)?\b",
            re.IGNORECASE,
        )

    def process(self, text: str) -> ParsedProductName:
        """Полная обработка названия."""
        if not text or not isinstance(text, str):
            return ParsedProductName(original=text or "", normalized="")
        cleaned = self._basic_cleaning(text)
        brand = self.extract_brand(cleaned)
        model_number = self.extract_model_number(cleaned)
        tokens = self.tokenize(cleaned)
        cleaned_tokens = self._process_tokens(tokens)
        normalized = " ".join(cleaned_tokens)
        return ParsedProductName(
            original=text,
            normalized=normalized,
            brand=brand,
            model_number=model_number,
            tokens=tokens,
            cleaned_tokens=cleaned_tokens,
        )

    def _basic_cleaning(self, text: str) -> str:
        result = text
        if REMOVE_CHARS:
            result = re.sub(REMOVE_CHARS, " ", result)
        result = re.sub(r"\s+", " ", result)
        if self.lowercase:
            result = result.lower()
        return result.strip()

    def extract_brand(self, text: str) -> Optional[str]:
        """Извлекает бренд."""
        match = self._brand_start_pattern.search(text)
        if match:
            brand_found = match.group(2) or match.group(1)
            if brand_found:
                return brand_found.lower().strip()
        text_lower = text.lower()
        for brand in KNOWN_BRANDS:
            if brand in text_lower:
                return brand
        return None

    def extract_model_number(self, text: str) -> Optional[str]:
        """Извлекает номер модели."""
        match = self._model_pattern.search(text)
        if match:
            return match.group(1).upper().strip()
        btu_match = self._btu_pattern.search(text)
        if btu_match:
            return f"BTU_{btu_match.group(1)}"
        return None

    def tokenize(self, text: str) -> List[str]:
        """Токенизирует текст."""
        tokens = re.split(r"[\s\-]+", text)
        return [token.strip() for token in tokens if token.strip()]

    def _process_tokens(self, tokens: List[str]) -> List[str]:
        result = []
        for token in tokens:
            if len(token) < 2 or len(token) > 30:
                continue
            if token.isdigit():
                continue
            processed_token = self._lemmatize(token) if self.use_lemmatization else token
            if processed_token not in STOP_WORDS_RU:
                result.append(processed_token)
        return result

    def _lemmatize(self, word: str) -> str:
        if not morph or len(word) < 3:
            return word
        try:
            parsed = morph.parse(word)
            if parsed:
                return parsed[0].normal_form()
        except Exception:
            pass
        return word

    @staticmethod
    def batch_process(texts: List[str]) -> List[ParsedProductName]:
        """Пакетная обработка названий."""
        processor = TextPreprocessor()
        return [processor.process(text) for text in texts]


class CharacteristicsNormalizer:
    """Нормализатор значений характеристик."""

    @staticmethod
    def normalize_numeric(value: str, feature_name: str = "") -> Optional[float]:
        """Нормализует числовое значение."""
        if not value or not isinstance(value, str):
            return None
        value = value.strip().replace(",", ".")
        value = re.sub(r"[^\d.\-]", "", value)
        if not value or value in {".", "-"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def normalize_boolean(value: str) -> Optional[bool]:
        """Нормализует булево значение."""
        if not value or not isinstance(value, str):
            return None
        value_lower = value.lower().strip()
        positive_values = {"да", "yes", "true", "1", "+", "есть"}
        negative_values = {"нет", "no", "false", "0", "-", "нету"}
        if value_lower in positive_values:
            return True
        if value_lower in negative_values:
            return False
        return None

    @staticmethod
    def parse_temperature_range(value: str) -> Tuple[Optional[float], Optional[float]]:
        """Парсит диапазон температур."""
        if not value or not isinstance(value, str):
            return None, None
        simple_pattern = re.compile(r"([-−]?\d+)\s*[-–]\s*([-−]?\d+)")
        match = simple_pattern.search(value)
        if match:
            return (
                CharacteristicsNormalizer.normalize_numeric(match.group(1)),
                CharacteristicsNormalizer.normalize_numeric(match.group(2)),
            )
        range_pattern = re.compile(r"от\s*([-−]?\d+)\s*(?:до|по)\s*([-−]?\d+)", re.IGNORECASE)
        match = range_pattern.search(value)
        if match:
            return (
                CharacteristicsNormalizer.normalize_numeric(match.group(1)),
                CharacteristicsNormalizer.normalize_numeric(match.group(2)),
            )
        single = CharacteristicsNormalizer.normalize_numeric(value)
        return single, single

    @staticmethod
    def parse_dimension(value: str) -> Dict[str, Optional[float]]:
        """Парсит размерность."""
        result = {"height": None, "width": None, "depth": None}
        if not value or not isinstance(value, str):
            return result
        pattern1 = re.compile(r"([\d.]+)\s*[×xX*]\s*([\d.]+)\s*[×xX*]\s*([\d.]+)")
        match = pattern1.search(value)
        if match:
            result["height"] = float(match.group(1))
            result["width"] = float(match.group(2))
            result["depth"] = float(match.group(3))
            return result
        pattern2 = re.compile(r"(\d+)\s*[xX*]\s*(\d+)\s*[xX*]\s*(\d+)")
        match = pattern2.search(value)
        if match:
            result["height"] = float(match.group(1)) / 1000
            result["width"] = float(match.group(2)) / 1000
            result["depth"] = float(match.group(3)) / 1000
        return result


def clean_product_data(product: Dict) -> Dict:
    """Очищает данные одного товара."""
    cleaned = {
        "id": str(product.get("id", "")),
        "name": product.get("Name", product.get("name", "")),
        "code": product.get("Code", product.get("code", "")),
    }
    if not cleaned["id"] and isinstance(product.get("_id"), dict):
        cleaned["id"] = str(product["_id"].get("$uuid", ""))
    characteristics = product.get("Characteristics", product.get("characteristics", []))
    parsed_chars: Dict[str, object] = {}
    if isinstance(characteristics, list):
        for char in characteristics:
            if not isinstance(char, dict):
                continue
            name = char.get("Name")
            value = char.get("Value")
            if name and value is not None:
                normalized_value = CharacteristicsNormalizer.normalize_numeric(str(value))
                parsed_chars[name] = normalized_value if normalized_value is not None else str(value).strip()
    elif isinstance(characteristics, dict):
        for name, value in characteristics.items():
            if value is None:
                continue
            normalized_value = CharacteristicsNormalizer.normalize_numeric(str(value))
            parsed_chars[name] = normalized_value if normalized_value is not None else str(value).strip()
    cleaned["characteristics"] = parsed_chars
    return cleaned
