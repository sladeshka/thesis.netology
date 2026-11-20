"""
Бизнес-слой системы подбора аналогов вентиляционного и климатического оборудования.
"""

import logging
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

EPS = 1e-8
WEIGHT_EPS = 1e-12
CRITICAL_IMPORTANCE = 3

NULLISH_TOKENS: Set[str] = {
    "",
    "nan",
    "none",
    "null",
    "-",
    "–",
    "—",
    "n/a",
    "na",
    "нет",
    "не указано",
    "не указан",
    "не заполнено",
    "unknown",
    "?",
}

BOOL_TRUE_EXACT: Set[str] = {
    "true",
    "yes",
    "y",
    "1",
    "on",
    "enabled",
    "available",
    "included",
    "integrated",
    "built-in",
    "builtin",
    "onboard",
    "on board",
    "да",
    "есть",
    "да есть",
    "есть да",
    "имеется",
    "имеет",
    "наличие",
    "присутствует",
    "доступно",
    "доступен",
    "доступна",
    "поддерживается",
    "поддерживает",
    "в наличии",
    "в комплекте",
    "в комплекте поставки",
    "в комплект",
    "встроен",
    "встроенный",
    "встроенная",
    "встроенное",
    "встроенном",
    "встроенной",
    "с функцией",
    "с модулем",
    "есть функция",
    "есть модуль",
}

BOOL_FALSE_EXACT: Set[str] = {
    "false",
    "no",
    "n",
    "0",
    "off",
    "disabled",
    "not available",
    "not included",
    "not supported",
    "missing",
    "absent",
    "off board",
    "offboard",
    "no wifi",
    "нет",
    "не",
    "отсутствует",
    "отсутств",
    "не имеется",
    "не имеет",
    "не имеет функции",
    "не поддерживает",
    "не поддерживается",
    "недоступно",
    "недоступен",
    "недоступна",
    "не в наличии",
    "не в комплекте",
    "не входит в комплект",
    "не входит в комплект поставки",
    "без",
    "без модуля",
    "без функции",
    "нет модуля",
    "нет функции",
    "отключено",
    "отключена",
    "отключён",
}

BOOL_NULLISH_EXACT: Set[str] = {
    "",
    "nan",
    "none",
    "null",
    "-",
    "–",
    "—",
    "n/a",
    "na",
    "нет данных",
    "нет информации",
    "не указано",
    "не указан",
    "не указана",
    "не заполнено",
    "не заполнен",
    "неизвестно",
    "неизвестен",
    "неизвестна",
    "не известно",
    "уточнить",
    "уточняется",
    "уточнить у продавца",
    "подтверждается",
    "unknown",
    "tbd",
    "to be determined",
    "under review",
}

BOOL_TRUE_PHRASES = [
    "в наличии",
    "в комплекте",
    "в комплекте поставки",
    "поддерживается",
    "поддерживает",
    "доступен",
    "доступна",
    "доступно",
    "имеется",
    "присутствует",
    "встроен",
    "встроенный",
    "встроенная",
    "встроенное",
    "built-in",
    "builtin",
    "on board",
    "onboard",
    "при подключении",
    "с функцией",
    "с модулем",
    "есть функция",
    "есть модуль",
    "есть",
    "да",
]

BOOL_FALSE_PHRASES = [
    "не в наличии",
    "не в комплекте",
    "не входит в комплект",
    "не входит в комплект поставки",
    "не поддерживается",
    "не поддерживает",
    "не имеет",
    "не имеется",
    "недоступно",
    "недоступен",
    "недоступна",
    "отсутствует",
    "отсутств",
    "нет модуля",
    "нет функции",
    "без модуля",
    "без функции",
    "off",
    "off board",
    "offboard",
    "no wifi",
    "не",
    "нет",
]

W_TO_KW = 1000.0

ENERGY_CLASS_SCORE_MAP: Dict[str, float] = {
    "A+++": 4.0,
    "A++": 3.0,
    "A+": 2.0,
    "A": 1.0,
    "B": 0.0,
    "C": -1.0,
    "D": -2.0,
    "E": -3.0,
    "F": -4.0,
    "G": -5.0,
}

STANDARD_MODEL_CLASS_KW: Dict[int, float] = {
    7: 2.0,
    9: 2.5,
    12: 3.5,
    18: 5.0,
    24: 7.0,
    28: 8.2,
    30: 8.8,
    36: 10.5,
    42: 12.0,
    48: 14.0,
    60: 17.5,
}

CORE_TECHNICAL_FIELDS: Set[str] = {
    "cooling_power_kw",
    "area_m2",
    "heating_power_kw",
}

DEFAULT_BUSINESS_SPECS: List[Dict[str, Any]] = [
    {
        "Name": "Инверторная технология",
        "Field": "is_inverter",
        "Kind": "bool",
        "Importance": 3,
        "TolerancePercent": None,
        "Unit": "",
    },
    {
        "Name": "Эффективен для помещ. площадью до",
        "Field": "area_m2",
        "Kind": "numeric_relative",
        "Importance": 3,
        "TolerancePercent": 15.0,
        "Unit": "м²",
    },
    {
        "Name": "Номинальная производительность охлаждения",
        "Field": "cooling_power_kw",
        "Kind": "numeric_relative",
        "Importance": 3,
        "TolerancePercent": 15.0,
        "Unit": "кВт",
    },
    {
        "Name": "Номинальная производительность обогрева",
        "Field": "heating_power_kw",
        "Kind": "numeric_relative",
        "Importance": 3,
        "TolerancePercent": 15.0,
        "Unit": "кВт",
    },
    {
        "Name": "Уровень шума внутр. блока",
        "Field": "noise_indoor_db",
        "Kind": "numeric_relative",
        "Importance": 3,
        "TolerancePercent": 40.0,
        "Unit": "дБ(А)",
    },
    {
        "Name": "Мин. рабочая температура воздуха для внешнего блока",
        "Field": "min_operating_temp_c",
        "Kind": "temp_relative",
        "Importance": 3,
        "TolerancePercent": 30.0,
        "Unit": "°C",
    },
    {
        "Name": "Цвет корпуса внутр. блока",
        "Field": "color_inner_body",
        "Kind": "category_exact",
        "Importance": 3,
        "TolerancePercent": None,
        "Unit": "",
    },
    {
        "Name": "Wi-Fi модуль",
        "Field": "has_wifi_module",
        "Kind": "bool",
        "Importance": 1,
        "TolerancePercent": None,
        "Unit": "",
    },
    {
        "Name": "Управление c мобильного приложения по Wi-Fi",
        "Field": "has_wifi_app_control",
        "Kind": "bool",
        "Importance": 1,
        "TolerancePercent": None,
        "Unit": "",
    },
    {
        "Name": "Хладагент",
        "Field": "refrigerant",
        "Kind": "category_exact",
        "Importance": 2,
        "TolerancePercent": None,
        "Unit": "",
    },
    {
        "Name": "Класс энергоэффективности",
        "Field": "energy_class",
        "Kind": "energy_class",
        "Importance": 2,
        "TolerancePercent": None,
        "Unit": "",
    },
]

CHAR_ALIASES: Dict[str, List[str]] = {
    "is_inverter": ["инверторная технология"],
    "area_m2": [
        "эффективен для помещ. площадью до",
        "площадь помещения до",
    ],
    "cooling_power_kw": [
        "номинальная производительность охлаждения",
        "макс. производительность охлаждения",
        "макс. производительность",
    ],
    "heating_power_kw": [
        "номинальная производительность обогрева",
        "макс. производительность обогрева",
    ],
    "noise_indoor_db": [
        "уровень шума внутр. блока",
        "уровень шума внутреннего блока",
    ],
    "min_operating_temp_c": [
        "мин. рабочая температура воздуха для внешнего блока",
        "мин. рабочая температура воздуха для наружного блока",
    ],
    "color_inner_body": [
        "цвет корпуса внутр. блока",
        "цвет корпуса внутреннего блока",
    ],
    "has_wifi_module": ["wi-fi модуль"],
    "has_wifi_app_control": [
        "управление c мобильного приложения по wi-fi",
        "управление с мобильного приложения по wi-fi",
    ],
    "refrigerant": ["хладагент"],
    "energy_class": [
        "класс энергоэффективности",
        "энергоэффективность",
    ],
}

COLOR_SYNONYMS_GROUPS: Dict[str, List[str]] = {
    "Белый": [
        "белый", "бел", "white", "белоснежный", "белоснежно-белый",
        "молочный", "молочно-белый", "молочный белый", "off white",
        "off-white", "бело-снежный", "bianco traffico", "ral 9016",
        "ral9016", "ral 016", "ral 9003", "ral9003", "ral 9010",
        "ral9010", "signal white",
    ],
    "Чёрный": [
        "черный", "чёрный", "black", "noir sable", "noir",
        "matte black", "матовый черный", "матовый чёрный",
        "черный матовый", "чёрный матовый", "ral 9005", "ral9005",
        "ral 9007", "ral9007", "jet black", "deep black",
    ],
    "Серебристый": [
        "серебристый", "silver", "silver satin", "серебро",
        "серебристый металлик", "ss", "/ss", "\\ss", "aluminium",
        "алюминий", "алюминиевый", "ral 9006", "ral9006",
        "silver metallic",
    ],
    "Серый": [
        "серый", "gray", "grey", "серебряный", "стальной", "сталь",
        "стальной цвет", "steel", "graphite", "графит", "графитовый",
        "металлик", "metallic", "металлический", "metal", "ral 7035",
        "ral7035", "ral 7040", "ral7040", "light gray", "light grey",
        "dark gray", "dark grey",
    ],
    "Слоновая кость": [
        "слоновая кость", "ivory", "ivoire", "ivory white",
        "белоснежная кость", "ral 1015", "ral1015",
    ],
    "Антрацит": [
        "антрацит", "anthracite", "anthracite grey",
        "anthracite gray", "ral 7016", "ral7016",
    ],
    "Бежевый": [
        "бежевый", "beige", "беж", "кремовый", "cream", "крем",
        "sand", "песочный", "песочный цвет", "camel", "камель",
        "tan", "тэн",
    ],
    "Коричневый": [
        "коричневый", "brown", "какао", "cocoa", "chocolate",
        "шоколадный", "coffee", "кофейный",
    ],
    "Бронзовый": [
        "бронзовый", "bronze", "бронза", "bronze metallic",
        "бронзовый металлик",
    ],
    "Золотой": [
        "золотой", "gold", "золотистый", "golden", "gold metallic",
        "золотой металлик", "золоченый", "shiny gold",
    ],
    "Синий": ["синий", "blue"],
    "Голубой": ["голубой", "light blue", "sky blue", "голубоватый"],
    "Красный": ["красный", "red", "алый", "crimson"],
    "Бордовый": ["бордовый", "burgundy", "wine", "вишнёвый", "cherry"],
    "Зелёный": [
        "зелёный", "зеленый", "green", "салатовый", "lime",
        "лимонно-зелёный", "emerald", "изумрудный",
    ],
    "Оранжевый": ["оранжевый", "orange", "персиковый", "peach"],
    "Розовый": ["розовый", "pink"],
    "Фиолетовый": ["фиолетовый", "purple", "пурпурный", "violet"],
    "Бирюзовый": ["бирюзовый", "turquoise", "aqua", "аqua"],
    "Титан": [
        "титан", "титаниум", "titanium", "titan", "titan gray",
        "titan grey", "titan silver",
    ],
    "Чёрно-белый": [
        "чёрно-белый", "черно-белый", "чёрный и белый",
        "черный и белый", "бело-чёрный", "бело-черный",
        "белый и черный", "белый и чёрный", "black white",
        "white black", "black/white", "white/black",
    ],
}

COLOR_RAL_CODES: Dict[str, str] = {
    "9016": "Белый",
    "9003": "Белый",
    "9010": "Белый",
    "9005": "Чёрный",
    "9007": "Чёрный",
    "9006": "Серебристый",
    "7035": "Серый",
    "7040": "Серый",
    "1015": "Слоновая кость",
    "7016": "Антрацит",
}

REFRIGERANT_SYNONYMS: Dict[str, str] = {
    "r410a": "R410A", "r-410a": "R410A", "r 410a": "R410A",
    "r410": "R410A", "r-410": "R410A", "r 410": "R410A",
    "410a": "R410A", "410": "R410A", "hfc-410a": "R410A",
    "hfc410a": "R410A", "hfc 410a": "R410A",
    "r32": "R32", "r-32": "R32", "r 32": "R32", "32": "R32",
    "hfc-32": "R32", "hfc32": "R32",
    "r22": "R22", "r-22": "R22", "r 22": "R22", "22": "R22",
    "r407c": "R407C", "r-407c": "R407C", "r 407c": "R407C",
    "407c": "R407C", "hfc-407c": "R407C", "hfc407c": "R407C",
    "r134a": "R134A", "r-134a": "R134A", "r 134a": "R134A",
    "134a": "R134A",
    "r290": "R290", "r-290": "R290", "r 290": "R290",
    "290": "R290",
    "r1234yf": "R1234YF", "r-1234yf": "R1234YF",
    "r 1234yf": "R1234YF", "1234yf": "R1234YF",
}

ENERGY_CLASS_SYNONYMS: Dict[str, str] = {
    "a+++": "A+++", "a +++": "A+++", "a+++/a+++": "A+++",
    "a++": "A++", "a ++": "A++", "a++/a++": "A++",
    "a++/a": "A++", "a+/a++": "A++",
    "a+": "A+", "a +": "A+", "a+/a": "A+", "a/a+": "A+",
    "a": "A", "a/a": "A",
    "b": "B", "b/b": "B",
    "c": "C", "c/c": "C",
    "d": "D", "d/d": "D",
    "e": "E", "e/e": "E",
    "f": "F", "f/f": "F",
    "g": "G", "g/g": "G",
}


# ---------------------------------------------------------------------
# Базовые текстовые и числовые утилиты
# ---------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


COLOR_SYNONYMS: Dict[str, str] = {}
for canonical, values in COLOR_SYNONYMS_GROUPS.items():
    for value in values:
        if value is None:
            continue
        key = normalize_text(value)
        if key:
            COLOR_SYNONYMS[key] = canonical


def is_nullish_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        token = normalize_text(value)
        if token in NULLISH_TOKENS:
            return True
        if not token:
            return True
    return False


def is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        num = float(value)
        return not math.isnan(num)
    except Exception:
        return False


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            num = float(value)
            return None if math.isnan(num) else num
        except Exception:
            return None

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace(",", ".")
    text = re.sub(r"(\d)\s+(?=\d)", r"\1", text)

    match = re.search(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    if not match:
        return None

    try:
        num = float(match.group(1))
        return None if math.isnan(num) else num
    except Exception:
        return None


def parse_bool_synonyms(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if is_nullish_value(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            num = float(value)
            if num == 1.0:
                return True
            if num == 0.0:
                return False
        except Exception:
            pass

    text = normalize_text(value)
    if not text:
        return None

    if text in BOOL_TRUE_EXACT:
        return True
    if text in BOOL_FALSE_EXACT:
        return False
    if text in BOOL_NULLISH_EXACT:
        return None

    for phrase in BOOL_FALSE_PHRASES:
        if phrase in text:
            return False

    for phrase in BOOL_TRUE_PHRASES:
        if phrase in text:
            return True

    return None


def parse_bool(value: Any) -> Optional[bool]:
    return parse_bool_synonyms(value)


def normalize_area_m2(raw: Any) -> Optional[float]:
    value = parse_number(raw)
    if value is None or value <= 0 or value > 2000:
        return None
    return float(value)


def normalize_power_kw(raw: Any) -> Optional[float]:
    value = parse_number(raw)
    if value is None or value <= 0:
        return None
    if value > 100:
        kw = value / W_TO_KW
    else:
        kw = value
    if kw <= 0 or kw > 250:
        return None
    return float(kw)


def normalize_noise_db(raw: Any) -> Optional[float]:
    value = parse_number(raw)
    if value is None or value <= 0 or value > 120:
        return None
    return float(value)


def normalize_temp_c(raw: Any) -> Optional[float]:
    value = parse_number(raw)
    if value is None or value < -80 or value > 80:
        return None
    return float(value)


def normalize_color_synonyms(value: Any) -> Optional[str]:
    if value is None or is_nullish_value(value):
        return None

    text = normalize_text(value)
    if not text:
        return None

    if text in COLOR_SYNONYMS:
        return COLOR_SYNONYMS[text]

    ral_match = re.search(r"ral\s*([0-9]{4})", text)
    if ral_match:
        code = ral_match.group(1)
        if code in COLOR_RAL_CODES:
            return COLOR_RAL_CODES[code]

    found_canonicals: List[str] = []
    sorted_keys = sorted(COLOR_SYNONYMS.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if not key:
            continue
        if key in text:
            canonical = COLOR_SYNONYMS[key]
            if canonical == "Чёрно-белый":
                return canonical
            if canonical not in found_canonicals:
                found_canonicals.append(canonical)

    if not found_canonicals:
        tokens = [t for t in re.split(r"\s+", text) if t]
        for token in tokens:
            if token in COLOR_SYNONYMS:
                return COLOR_SYNONYMS[token]

        clean = re.sub(r"[^a-zа-яё0-9\s\-/]+", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return None
        return clean.title()

    if len(found_canonicals) == 1:
        return found_canonicals[0]

    return "Смешанный"


def normalize_color(value: Any) -> Optional[str]:
    return normalize_color_synonyms(value)


def normalize_refrigerant_synonyms(value: Any) -> Optional[str]:
    if value is None or is_nullish_value(value):
        return None

    text = normalize_text(value)
    if not text:
        return None

    compact = re.sub(r"[\s\-]+", "", text)
    if compact in REFRIGERANT_SYNONYMS:
        return REFRIGERANT_SYNONYMS[compact]

    if text in REFRIGERANT_SYNONYMS:
        return REFRIGERANT_SYNONYMS[text]

    match = re.search(r"r[- ]?(\d+[a-z]?)", text)
    if match:
        candidate = "R" + match.group(1).upper()
        if candidate in {"R410", "R410A", "R410a"}:
            return "R410A"
        if candidate in REFRIGERANT_SYNONYMS.values():
            return candidate

    return text.upper()


def normalize_category(value: Any, max_length: int = 80) -> Optional[str]:
    if is_nullish_value(value):
        return None

    parsed_bool = parse_bool_synonyms(value)
    if parsed_bool is not None:
        return "true" if parsed_bool else "false"

    text = normalize_text(value)
    if not text or is_nullish_value(text):
        return None

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[:max_length].strip()

    return text or None


def normalize_energy_class_synonyms(value: Any) -> Optional[str]:
    if value is None or is_nullish_value(value):
        return None

    text = normalize_text(value)
    if not text:
        return None

    if text in ENERGY_CLASS_SYNONYMS:
        return ENERGY_CLASS_SYNONYMS[text]

    compact = re.sub(r"\s+", "", text)
    if compact in ENERGY_CLASS_SYNONYMS:
        return ENERGY_CLASS_SYNONYMS[compact]

    match = re.search(r"([a-g])\s*(\+*)", text)
    if match:
        letter = match.group(1).upper()
        plus = match.group(2)
        return letter + plus

    return None


def normalize_energy_class(value: Any) -> Optional[str]:
    return normalize_energy_class_synonyms(value)


def energy_class_to_score(value: Any) -> Optional[float]:
    if value is None or is_nullish_value(value):
        return None

    canonical = normalize_energy_class_synonyms(value)
    if canonical is None:
        text = normalize_text(value)
        if not text:
            return None

        compact = re.sub(r"[^a-z+]", "", text)
        matches = re.findall(r"([a-g])(\+*)", compact)
        if not matches:
            return None

        letter, plus = matches[0]
        canonical = letter.upper() + plus

    canonical = canonical.upper()
    if canonical in ENERGY_CLASS_SCORE_MAP:
        return ENERGY_CLASS_SCORE_MAP[canonical]

    match = re.fullmatch(r"([A-G])(\+{0,3})", canonical)
    if not match:
        return None

    return ENERGY_CLASS_SCORE_MAP.get(canonical)


# ---------------------------------------------------------------------
# Бизнес-контекст
# ---------------------------------------------------------------------

@dataclass
class BusinessRuleContext:
    specs: List[Dict[str, Any]]
    weights: Dict[str, float]
    field_to_kind: Dict[str, str]
    field_to_tolerance: Dict[str, Optional[float]]
    field_to_importance: Dict[str, int]
    field_to_name: Dict[str, str]
    importance_weight: Dict[int, float]


def build_business_context(
    specs: Optional[List[Dict[str, Any]]] = None,
) -> BusinessRuleContext:
    if specs is None:
        specs = DEFAULT_BUSINESS_SPECS

    importance_weight = {
        0: 0.00,
        1: 0.25,
        2: 0.75,
        3: 1.00,
    }

    raw_weights: Dict[str, float] = {}
    for spec in specs:
        weight = importance_weight[int(spec["Importance"])]
        if weight > 0:
            raw_weights[spec["Field"]] = weight

    total_weight = sum(raw_weights.values())
    weights = {
        field_name: float(weight / total_weight)
        for field_name, weight in raw_weights.items()
    }

    return BusinessRuleContext(
        specs=specs,
        weights=weights,
        field_to_kind={spec["Field"]: spec["Kind"] for spec in specs},
        field_to_tolerance={spec["Field"]: spec.get("TolerancePercent") for spec in specs},
        field_to_importance={spec["Field"]: int(spec["Importance"]) for spec in specs},
        field_to_name={spec["Field"]: spec["Name"] for spec in specs},
        importance_weight=importance_weight,
    )


# ---------------------------------------------------------------------
# Сравнение компонентов
# ---------------------------------------------------------------------

def absolute_difference(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if not is_valid_number(a) or not is_valid_number(b):
        return None
    return abs(float(a) - float(b))


def compare_numeric_relative(
    query_value: Any,
    candidate_value: Any,
    tolerance_percent: Optional[float],
    importance: int = 0,
) -> Tuple[float, bool, str]:
    if not is_valid_number(query_value) or not is_valid_number(candidate_value):
        return 0.5, False, "Нет данных для сравнения числового признака"

    q = float(query_value)
    c = float(candidate_value)
    diff = absolute_difference(q, c)
    if diff is None:
        return 0.5, False, "Ошибка сравнения числового признака"

    reference = max(abs(q), 1.0)
    if tolerance_percent is None:
        allowed = 0.0
    else:
        allowed = reference * float(tolerance_percent) / 100.0

    if diff <= allowed:
        return 1.0, False, "В пределах допуска"

    score = max(
        0.0,
        1.0 - (diff - allowed) / max(allowed, reference * 0.2, EPS)
    )

    reason = (
        f"Отклонение {diff:.3g} превышает допуск {allowed:.3g}"
        f" ({float(tolerance_percent or 0):.0f}%)"
    )
    return score, False, reason


def compare_temp_relative(
    query_value: Any,
    candidate_value: Any,
    tolerance_percent: Optional[float],
    importance: int = 0,
) -> Tuple[float, bool, str]:
    if not is_valid_number(query_value) or not is_valid_number(candidate_value):
        return 0.5, False, "Нет данных для сравнения температуры"

    q = float(query_value)
    c = float(candidate_value)
    diff = absolute_difference(q, c)
    if diff is None:
        return 0.5, False, "Ошибка сравнения температуры"

    reference = max(abs(q), 1.0)
    if tolerance_percent is None:
        allowed = 0.0
    else:
        allowed = reference * float(tolerance_percent) / 100.0

    if diff <= allowed:
        return 1.0, False, "Температура в пределах допуска"

    reason = (
        f"Слишком большое отличие температуры: {diff:.3g} > {allowed:.3g}"
    )
    return 0.0, True, reason


def compare_bool(
    query_value: Any,
    candidate_value: Any,
    importance: int,
) -> Tuple[float, bool, str]:
    q_valid = not is_nullish_value(query_value)
    c_valid = not is_nullish_value(candidate_value)

    if not q_valid or not c_valid:
        return 0.5, False, "Нет данных по булевому признаку"

    q = query_value is True
    c = candidate_value is True

    if q == c:
        return 1.0, False, "Совпадение булевого признака"

    hard_fail = importance >= CRITICAL_IMPORTANCE
    reason = (
        "Критическое несоответствие булевого признака"
        if hard_fail
        else "Несовпадение булевого признака"
    )
    return 0.0, hard_fail, reason


def compare_category_exact(
    query_value: Any,
    candidate_value: Any,
    importance: int,
) -> Tuple[float, bool, str]:
    q_valid = not is_nullish_value(query_value)
    c_valid = not is_nullish_value(candidate_value)

    if not q_valid or not c_valid:
        return 0.5, False, "Нет данных по категориальному признаку"

    q = str(query_value).strip().lower()
    c = str(candidate_value).strip().lower()

    if q == c:
        return 1.0, False, "Категория совпадает"

    hard_fail = importance >= CRITICAL_IMPORTANCE
    reason = (
        "Критическое несоответствие категории"
        if hard_fail
        else "Несовпадение категории"
    )
    return 0.0, hard_fail, reason


def compare_energy_class(
    query_value: Any,
    candidate_value: Any,
    importance: int = 2,
) -> Tuple[float, bool, str]:
    q_score = energy_class_to_score(query_value)
    c_score = energy_class_to_score(candidate_value)

    if q_score is None or c_score is None:
        return 0.5, False, "Нет данных по классу энергоэффективности"

    if c_score >= q_score:
        return 1.0, False, "Класс энергоэффективности не хуже исходного"

    diff = q_score - c_score
    score = max(0.0, 1.0 - diff / 4.0)
    return score, False, f"Класс энергоэффективности хуже на {diff:.1f}"


def is_present_for_field(kind: str, query_value: Any, candidate_value: Any) -> bool:
    if kind in {"numeric_relative", "temp_relative"}:
        return is_valid_number(query_value) and is_valid_number(candidate_value)
    return not is_nullish_value(query_value) and not is_nullish_value(candidate_value)


# ---------------------------------------------------------------------
# Основной бизнес-score
# ---------------------------------------------------------------------

def business_pair_score(
    query_row: pd.Series,
    candidate_row: pd.Series,
    context: BusinessRuleContext,
) -> Dict[str, Any]:
    total_weight = 0.0
    valid_weight = 0.0
    weighted_sum = 0.0

    hard_fail = False
    active_fields: List[str] = []
    valid_fields: List[str] = []
    missing_fields: List[str] = []
    hard_fail_fields: List[str] = []
    reasons: List[str] = []
    details: Dict[str, Dict[str, Any]] = {}

    for spec in context.specs:
        field = spec["Field"]
        kind = spec["Kind"]
        importance = int(spec["Importance"])
        tolerance = spec.get("TolerancePercent")
        weight = context.weights.get(field, 0.0)

        if weight <= WEIGHT_EPS:
            details[field] = {
                "score": 0.0,
                "component_score": 0.0,
                "weight": 0.0,
                "hard_fail": False,
                "skipped": True,
                "missing": False,
                "importance": importance,
                "kind": kind,
                "display_name": spec["Name"],
                "reason": "zero_weight",
            }
            continue

        qv = query_row.get(field)
        cv = candidate_row.get(field)
        present = is_present_for_field(kind, qv, cv)

        if not present:
            component_score, fail, reason = 0.5, False, "Нет данных"
            missing = True
        else:
            missing = False
            if kind == "numeric_relative":
                component_score, fail, reason = compare_numeric_relative(
                    qv, cv, tolerance, importance
                )
            elif kind == "temp_relative":
                component_score, fail, reason = compare_temp_relative(
                    qv, cv, tolerance, importance
                )
            elif kind == "bool":
                component_score, fail, reason = compare_bool(qv, cv, importance)
            elif kind == "category_exact":
                component_score, fail, reason = compare_category_exact(
                    qv, cv, importance
                )
            elif kind == "energy_class":
                component_score, fail, reason = compare_energy_class(
                    qv, cv, importance
                )
            else:
                component_score, fail, reason = 0.5, False, "Неизвестный тип признака"

        if fail and importance < CRITICAL_IMPORTANCE:
            fail = False

        total_weight += weight
        active_fields.append(field)

        if fail:
            hard_fail = True
            hard_fail_fields.append(field)
            reasons.append(f"{spec['Name']}: {reason}")

        if missing:
            missing_fields.append(field)
        else:
            valid_fields.append(field)
            valid_weight += weight
            weighted_sum += weight * component_score
            if component_score < 1.0 and reason != "Нет данных":
                reasons.append(f"{spec['Name']}: {reason}")

        details[field] = {
            "score": float(component_score),
            "component_score": float(component_score),
            "weight": float(weight),
            "hard_fail": bool(fail),
            "skipped": False,
            "missing": bool(missing),
            "importance": importance,
            "kind": kind,
            "display_name": spec["Name"],
            "reason": reason,
        }

    if total_weight <= WEIGHT_EPS:
        return {
            "score": 0.5,
            "valid_score": None,
            "total_weight": 0.0,
            "valid_weight": 0.0,
            "confidence": 0.0,
            "hard_fail": False,
            "status": "no_business_features",
            "reason": "no_business_features",
            "active_fields": active_fields,
            "valid_fields": valid_fields,
            "missing_fields": missing_fields,
            "hard_fail_fields": hard_fail_fields,
            "component_details": details,
            "reasons": reasons[:5],
            "valid_field_count": 0,
            "core_valid_field_count": 0,
        }

    confidence = valid_weight / total_weight

    if valid_weight > 0:
        score = weighted_sum / total_weight
        valid_score = weighted_sum / valid_weight
    else:
        score = 0.5
        valid_score = None

    if valid_weight <= WEIGHT_EPS:
        status = "no_valid_feature_data"
        reason = "no_feature_value_present_for_both_products"
    else:
        status = "ok"
        reason = None

    core_valid_field_count = len([f for f in valid_fields if f in CORE_TECHNICAL_FIELDS])

    return {
        "score": float(score),
        "valid_score": float(valid_score) if valid_score is not None else None,
        "total_weight": float(total_weight),
        "valid_weight": float(valid_weight),
        "confidence": float(confidence),
        "hard_fail": bool(hard_fail),
        "status": status,
        "reason": reason,
        "active_fields": active_fields,
        "valid_fields": valid_fields,
        "missing_fields": missing_fields,
        "hard_fail_fields": hard_fail_fields,
        "component_details": details,
        "reasons": reasons[:5],
        "valid_field_count": len(valid_fields),
        "core_valid_field_count": core_valid_field_count,
    }


def compute_pair_validity(
    query_row: pd.Series,
    candidate_row: pd.Series,
    context: BusinessRuleContext,
) -> Dict[str, Any]:
    valid_fields: List[str] = []
    missing_fields: List[str] = []
    core_valid_fields: List[str] = []
    core_available_fields: List[str] = []

    total_weight = 0.0
    valid_weight = 0.0
    core_available_weight = 0.0
    core_valid_weight = 0.0

    for spec in context.specs:
        field = spec["Field"]
        kind = spec["Kind"]
        weight = context.weights.get(field, 0.0)

        if weight <= WEIGHT_EPS:
            continue

        qv = query_row.get(field)
        cv = candidate_row.get(field)
        present = is_present_for_field(kind, qv, cv)

        total_weight += weight

        if field in CORE_TECHNICAL_FIELDS:
            core_available_fields.append(field)
            core_available_weight += weight

        if present:
            valid_fields.append(field)
            valid_weight += weight

            if field in CORE_TECHNICAL_FIELDS:
                core_valid_fields.append(field)
                core_valid_weight += weight
        else:
            missing_fields.append(field)

    confidence = valid_weight / total_weight if total_weight > 0 else 0.0
    core_confidence = (
        core_valid_weight / core_available_weight
        if core_available_weight > 0
        else 0.0
    )

    required_core_count = (
        min(2, len(core_available_fields)) if core_available_fields else 0
    )

    is_evaluable = (
        len(valid_fields) >= 2
        and len(core_valid_fields) >= required_core_count
    )

    return {
        "is_evaluable": bool(is_evaluable),
        "reason": None if is_evaluable else "insufficient_technical_data",
        "valid_fields": valid_fields,
        "missing_fields": missing_fields,
        "core_valid_fields": core_valid_fields,
        "valid_field_count": len(valid_fields),
        "core_valid_field_count": len(core_valid_fields),
        "core_required_field_count": int(required_core_count),
        "total_weight": float(total_weight),
        "valid_weight": float(valid_weight),
        "confidence": float(confidence),
        "core_confidence": float(core_confidence),
    }


# ---------------------------------------------------------------------
# Диагностика поля Analogs
# ---------------------------------------------------------------------

def build_analog_ground_truth(df: pd.DataFrame) -> Dict[str, Set[str]]:
    valid_ids = set(df["id"].astype(str).tolist())
    ground_truth: Dict[str, Set[str]] = {}

    for row in df.itertuples(index=False):
        query_id = str(row.id)
        raw_analogs = getattr(row, "analogs", [])
        analog_ids = set()

        for analog_id in raw_analogs:
            analog_id = str(analog_id).strip()
            if not analog_id:
                continue
            if analog_id == query_id:
                continue
            if analog_id not in valid_ids:
                continue
            analog_ids.add(analog_id)

        if analog_ids:
            ground_truth[query_id] = analog_ids

    return ground_truth


def build_evaluable_ground_truth(
    df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    context: BusinessRuleContext,
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    df_indexed = df.set_index("id", drop=False)
    evaluable_ground_truth: Dict[str, Set[str]] = {}

    total_pairs = 0
    evaluable_pairs = 0
    uncheckable_pairs = 0

    examples_evaluable: List[Dict[str, Any]] = []
    examples_uncheckable: List[Dict[str, Any]] = []

    for query_id, analog_ids in ground_truth.items():
        if query_id not in df_indexed.index:
            continue

        query_row = df_indexed.loc[query_id]
        if isinstance(query_row, pd.DataFrame):
            query_row = query_row.iloc[0]

        valid_analogs: Set[str] = set()

        for analog_id in analog_ids:
            if analog_id not in df_indexed.index:
                continue

            candidate_row = df_indexed.loc[analog_id]
            if isinstance(candidate_row, pd.DataFrame):
                candidate_row = candidate_row.iloc[0]

            validity = compute_pair_validity(query_row, candidate_row, context)
            total_pairs += 1

            if validity["is_evaluable"]:
                evaluable_pairs += 1
                valid_analogs.add(analog_id)
                if len(examples_evaluable) < 10:
                    examples_evaluable.append(
                        {
                            "query_id": query_id,
                            "analog_id": analog_id,
                            "valid_fields": validity["valid_fields"],
                            "core_valid_fields": validity["core_valid_fields"],
                            "confidence": validity["confidence"],
                        }
                    )
            else:
                uncheckable_pairs += 1
                if len(examples_uncheckable) < 20:
                    examples_uncheckable.append(
                        {
                            "query_id": query_id,
                            "analog_id": analog_id,
                            "valid_fields": validity["valid_fields"],
                            "missing_fields": validity["missing_fields"],
                            "confidence": validity["confidence"],
                        }
                    )

        if valid_analogs:
            evaluable_ground_truth[query_id] = valid_analogs

    stats = {
        "total_pairs": int(total_pairs),
        "evaluable_pairs": int(evaluable_pairs),
        "uncheckable_pairs": int(uncheckable_pairs),
        "evaluable_rate": float(evaluable_pairs / max(total_pairs, 1)),
        "queries_total": int(len(ground_truth)),
        "queries_with_evaluable_analogs": int(len(evaluable_ground_truth)),
        "examples_uncheckable": examples_uncheckable,
        "examples_evaluable": examples_evaluable,
    }

    return evaluable_ground_truth, stats


def validate_analogs_with_business_rules(
    df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    context: BusinessRuleContext,
    threshold: float = 0.60,
    min_confidence: float = 0.50,
) -> Dict[str, Any]:
    df_indexed = df.set_index("id", drop=False)

    total_checked = 0
    passed = 0
    failed_hard_fail = 0
    failed_score = 0
    failed_no_valid_feature_data = 0
    failed_low_confidence = 0

    confidence_values: List[float] = []
    score_values: List[float] = []
    hard_fail_field_counts: Dict[str, int] = {}
    failed_examples: List[Dict[str, Any]] = []

    for query_id, analog_ids in ground_truth.items():
        if query_id not in df_indexed.index:
            continue

        query_row = df_indexed.loc[query_id]
        if isinstance(query_row, pd.DataFrame):
            query_row = query_row.iloc[0]

        for analog_id in analog_ids:
            if analog_id not in df_indexed.index:
                continue

            candidate_row = df_indexed.loc[analog_id]
            if isinstance(candidate_row, pd.DataFrame):
                candidate_row = candidate_row.iloc[0]

            scoring = business_pair_score(query_row, candidate_row, context)
            total_checked += 1

            confidence = float(scoring.get("confidence", 0.0))
            score = float(scoring.get("score", 0.0))
            status = scoring.get("status")

            confidence_values.append(confidence)
            score_values.append(score)

            if status in {"no_business_features", "no_valid_feature_data"}:
                failed_no_valid_feature_data += 1
                if len(failed_examples) < 30:
                    failed_examples.append(
                        {
                            "query_id": query_id,
                            "analog_id": analog_id,
                            "failure_type": "insufficient_data",
                            "status": status,
                            "reason": scoring.get("reason"),
                            "score": score,
                            "confidence": confidence,
                            "missing_fields": scoring.get("missing_fields", []),
                        }
                    )
                continue

            if confidence < min_confidence:
                failed_low_confidence += 1
                if len(failed_examples) < 30:
                    failed_examples.append(
                        {
                            "query_id": query_id,
                            "analog_id": analog_id,
                            "failure_type": "low_confidence",
                            "score": score,
                            "confidence": confidence,
                            "missing_fields": scoring.get("missing_fields", []),
                        }
                    )
                continue

            if scoring.get("hard_fail"):
                failed_hard_fail += 1
                for field in scoring.get("hard_fail_fields", []):
                    hard_fail_field_counts[field] = hard_fail_field_counts.get(field, 0) + 1

                if len(failed_examples) < 30:
                    failed_examples.append(
                        {
                            "query_id": query_id,
                            "analog_id": analog_id,
                            "failure_type": "hard_fail",
                            "hard_fail_fields": scoring.get("hard_fail_fields", []),
                            "score": score,
                            "confidence": confidence,
                        }
                    )
                continue

            if score >= threshold:
                passed += 1
            else:
                failed_score += 1
                if len(failed_examples) < 30:
                    failed_examples.append(
                        {
                            "query_id": query_id,
                            "analog_id": analog_id,
                            "failure_type": "score_below_threshold",
                            "score": score,
                            "confidence": confidence,
                            "missing_fields": scoring.get("missing_fields", []),
                        }
                    )

    pass_rate = passed / max(total_checked, 1)
    evaluable_total = (
        total_checked
        - failed_no_valid_feature_data
        - failed_low_confidence
    )

    return {
        "threshold": float(threshold),
        "min_confidence": float(min_confidence),
        "total_checked": int(total_checked),
        "passed": int(passed),
        "failed": int(total_checked - passed),
        "failed_hard_fail": int(failed_hard_fail),
        "failed_score": int(failed_score),
        "failed_no_valid_feature_data": int(failed_no_valid_feature_data),
        "failed_low_confidence": int(failed_low_confidence),
        "evaluable_total": int(evaluable_total),
        "evaluable_rate": float(evaluable_total / max(total_checked, 1)),
        "pass_rate": float(pass_rate),
        "pass_rate_evaluable": float(passed / max(evaluable_total, 1)),
        "mean_score": float(np.mean(score_values)) if score_values else 0.0,
        "mean_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "hard_fail_field_counts": hard_fail_field_counts,
        "failed_examples": failed_examples[:30],
    }

def compute_analogs_source_quality(
    df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    context: BusinessRuleContext,
    threshold: float = 0.60,
) -> Dict[str, Any]:
    evaluable_ground_truth, evaluable_stats = build_evaluable_ground_truth(
        df=df,
        ground_truth=ground_truth,
        context=context,
    )

    all_pairs_validation = validate_analogs_with_business_rules(
        df=df,
        ground_truth=ground_truth,
        context=context,
        threshold=threshold,
        min_confidence=0.0,
    )

    evaluable_pairs_validation = validate_analogs_with_business_rules(
        df=df,
        ground_truth=evaluable_ground_truth,
        context=context,
        threshold=threshold,
        min_confidence=0.0,
    )

    total_pairs = evaluable_stats["total_pairs"]
    evaluable_pairs = evaluable_stats["evaluable_pairs"]
    uncheckable_pairs = evaluable_stats["uncheckable_pairs"]

    passed_all = all_pairs_validation["passed"]
    passed_evaluable = evaluable_pairs_validation["passed"]

    metrics = {
        "threshold": float(threshold),
        "total_pairs": int(total_pairs),
        "evaluable_pairs": int(evaluable_pairs),
        "uncheckable_pairs": int(uncheckable_pairs),
        "evaluable_rate": float(evaluable_pairs / max(total_pairs, 1)),
        "uncheckable_rate": float(uncheckable_pairs / max(total_pairs, 1)),
        "passed_all": int(passed_all),
        "pass_rate_all": float(passed_all / max(total_pairs, 1)),
        "passed_evaluable": int(passed_evaluable),
        "pass_rate_evaluable": float(passed_evaluable / max(evaluable_pairs, 1)),
        "failed_score_all": int(all_pairs_validation["failed_score"]),
        "failed_hard_fail_all": int(all_pairs_validation["failed_hard_fail"]),
        "failed_no_valid_feature_data_all": int(all_pairs_validation["failed_no_valid_feature_data"]),
        "failed_low_confidence_all": int(all_pairs_validation["failed_low_confidence"]),
        "failed_score_evaluable": int(evaluable_pairs_validation["failed_score"]),
        "failed_hard_fail_evaluable": int(evaluable_pairs_validation["failed_hard_fail"]),
        "failed_no_valid_feature_data_evaluable": int(evaluable_pairs_validation["failed_no_valid_feature_data"]),
        "failed_low_confidence_evaluable": int(evaluable_pairs_validation["failed_low_confidence"]),
        "mean_score_all": float(all_pairs_validation["mean_score"]),
        "mean_confidence_all": float(all_pairs_validation["mean_confidence"]),
        "mean_score_evaluable": float(evaluable_pairs_validation["mean_score"]),
        "mean_confidence_evaluable": float(evaluable_pairs_validation["mean_confidence"]),
    }

    return {
        "source_quality_metrics": metrics,
        "evaluable_ground_truth": evaluable_ground_truth,
        "all_pairs_validation": {
            key: value for key, value in all_pairs_validation.items()
            if key != "failed_examples"
        },
        "evaluable_pairs_validation": {
            key: value for key, value in evaluable_pairs_validation.items()
            if key != "failed_examples"
        },
    }


# ---------------------------------------------------------------------
# Построение профиля товара из JSON
# ---------------------------------------------------------------------

def build_characteristic_map(record: Dict[str, Any]) -> Dict[str, Any]:
    chars = record.get("Characteristics", record.get("characteristics", []))
    char_map: Dict[str, Any] = {}

    if isinstance(chars, list):
        for item in chars:
            if not isinstance(item, dict):
                continue

            name = item.get("Name", item.get("name", ""))
            value = item.get("Value", item.get("value"))

            if not name:
                continue

            key = normalize_text(name)
            if not key:
                continue

            if value is None:
                continue

            if key not in char_map:
                char_map[key] = value
    elif isinstance(chars, dict):
        for name, value in chars.items():
            if not name:
                continue
            key = normalize_text(name)
            if key and value is not None and key not in char_map:
                char_map[key] = value

    return char_map


def first_non_null_by_aliases(
    char_map: Dict[str, Any],
    aliases: List[str],
) -> Optional[Any]:
    for alias in aliases:
        value = char_map.get(normalize_text(alias))
        if not is_nullish_value(value):
            return value
    return None


def normalize_analog_ids(analog_ids: Iterable[Any], self_id: str) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()

    for analog_id in analog_ids or []:
        pid = str(analog_id).strip()
        if not pid:
            continue
        if pid == self_id:
            continue
        if pid not in seen:
            result.append(pid)
            seen.add(pid)

    return result


def get_raw_analog_ids(record: Dict[str, Any]) -> List[str]:
    analogs = record.get("Analogs", record.get("analogs", []))
    result: List[str] = []

    if isinstance(analogs, list):
        for item in analogs:
            if isinstance(item, dict):
                pid = item.get("ProductId", item.get("Id", item.get("id", "")))
            else:
                pid = item

            pid = str(pid).strip()
            if pid:
                result.append(pid)

    return result


def build_product_profile(product_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(product_record, dict):
        return None

    product_id = product_record.get("Id", product_record.get("id", ""))
    product_id = str(product_id).strip()

    name = product_record.get("Name", product_record.get("name", ""))
    code = product_record.get("Code", product_record.get("code", ""))

    if not product_id:
        raw_key = f"{name or ''}|{code or ''}"
        product_id = f"fallback_{uuid.uuid5(uuid.NAMESPACE_DNS, raw_key)}"

    char_map = build_characteristic_map(product_record)

    profile: Dict[str, Any] = {
        "id": product_id,
        "name": name,
        "normalized_name": normalize_text(name),
        "code": str(code).strip(),
        "analogs": normalize_analog_ids(get_raw_analog_ids(product_record), product_id),
    }

    numeric_parsers = {
        "area_m2": normalize_area_m2,
        "cooling_power_kw": normalize_power_kw,
        "heating_power_kw": normalize_power_kw,
        "noise_indoor_db": normalize_noise_db,
        "min_operating_temp_c": normalize_temp_c,
    }

    bool_fields = {
        "is_inverter",
        "has_wifi_module",
        "has_wifi_app_control",
    }

    for field_name, aliases in CHAR_ALIASES.items():
        raw_value = first_non_null_by_aliases(char_map, aliases)

        if field_name == "color_inner_body":
            profile[field_name] = normalize_color(raw_value)
        elif field_name in bool_fields:
            profile[field_name] = parse_bool(raw_value)
        elif field_name == "refrigerant":
            profile[field_name] = normalize_refrigerant_synonyms(raw_value)
        elif field_name == "energy_class":
            profile[field_name] = normalize_energy_class(raw_value)
        else:
            parser = numeric_parsers.get(field_name)
            profile[field_name] = parser(raw_value) if parser else None

    return profile


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, bool):
        return False

    if isinstance(value, (list, dict, set, frozenset)):
        return len(value) == 0

    if isinstance(value, str):
        return is_nullish_value(value)

    try:
        return math.isnan(float(value))
    except Exception:
        return False


def unique_str_list(items: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()

    for item in items or []:
        value = str(item).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)

    return result


def merge_duplicate_profiles(profiles: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for raw_profile in profiles:
        profile = dict(raw_profile)
        product_id = str(profile.get("id", "")).strip()

        if not product_id:
            continue

        profile["analogs"] = normalize_analog_ids(profile.get("analogs", []), product_id)

        if product_id not in merged:
            item = dict(profile)

            name_variants: List[str] = []
            if not is_empty_value(item.get("name")):
                name_variants.append(str(item.get("name")).strip())
            if not is_empty_value(item.get("normalized_name")):
                name_variants.append(str(item.get("normalized_name")).strip())

            item["name_variants"] = unique_str_list(name_variants)
            item["normalized_name_variants"] = unique_str_list(
                [normalize_text(x) for x in item["name_variants"]]
            )
            item["name_haystack"] = " ".join(item["normalized_name_variants"]).lower()
            item["_merge_conflicts"] = []

            merged[product_id] = item
            continue

        target = merged[product_id]

        existing_analogs = normalize_analog_ids(target.get("analogs", []), product_id)
        seen_analogs = set(existing_analogs)

        for analog_id in profile.get("analogs", []):
            analog_id = str(analog_id).strip()
            if not analog_id or analog_id == product_id:
                continue
            if analog_id not in seen_analogs:
                existing_analogs.append(analog_id)
                seen_analogs.add(analog_id)

        target["analogs"] = existing_analogs

        incoming_names: List[str] = []
        if not is_empty_value(profile.get("name")):
            incoming_names.append(str(profile.get("name")).strip())
        if not is_empty_value(profile.get("normalized_name")):
            incoming_names.append(str(profile.get("normalized_name")).strip())

        target["name_variants"] = unique_str_list(
            target.get("name_variants", []) + incoming_names
        )
        target["normalized_name_variants"] = unique_str_list(
            [normalize_text(x) for x in target.get("name_variants", [])]
        )
        target["name_haystack"] = " ".join(target.get("normalized_name_variants", [])).lower()

        if is_empty_value(target.get("name")):
            target["name"] = profile.get("name")
        if is_empty_value(target.get("normalized_name")):
            target["normalized_name"] = profile.get("normalized_name")

        skip_fields = {
            "id",
            "name",
            "normalized_name",
            "name_variants",
            "normalized_name_variants",
            "name_haystack",
            "analogs",
            "_merge_conflicts",
        }

        for field_name, incoming_value in profile.items():
            if field_name in skip_fields:
                continue

            existing_value = target.get(field_name)

            if is_empty_value(existing_value):
                target[field_name] = incoming_value
                continue

            conflict = False

            if isinstance(existing_value, float) and isinstance(incoming_value, float):
                if abs(existing_value - incoming_value) >= 1e-6:
                    conflict = True
            else:
                if str(existing_value).lower() != str(incoming_value).lower():
                    conflict = True

            if conflict:
                target.setdefault("_merge_conflicts", []).append(
                    {
                        "field": field_name,
                        "kept": existing_value,
                        "incoming": incoming_value,
                    }
                )

    return list(merged.values())


def build_profiles_from_records(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    raw_profiles: List[Dict[str, Any]] = []
    validation_errors = 0

    for idx, record in enumerate(records):
        try:
            profile = build_product_profile(record)
            if profile is not None:
                raw_profiles.append(profile)
        except Exception as exc:
            validation_errors += 1
            if validation_errors <= 20:
                LOGGER.warning("Ошибка построения профиля на записи %d: %s", idx, exc)

    merged_profiles = merge_duplicate_profiles(raw_profiles)
    df = pd.DataFrame(merged_profiles)

    LOGGER.info(
        "Профили построены: raw=%d, merged=%d, validation_errors=%d",
        len(raw_profiles),
        len(df),
        validation_errors,
    )

    return df


def filter_by_name_keywords(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    if not keywords:
        return df.copy()

    if "name_haystack" in df.columns:
        text = df["name_haystack"].fillna("").astype(str).str.lower()
    elif "normalized_name" in df.columns:
        text = df["normalized_name"].fillna("").astype(str).str.lower()
    else:
        text = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()

    mask = pd.Series(False, index=df.index)

    for keyword in keywords:
        mask |= text.str.contains(str(keyword).lower(), na=False, regex=False)

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------
# Текстовая близость и candidate generation
# ---------------------------------------------------------------------

def _analog_numeric_similarity_series(
    cand: pd.DataFrame,
    field: str,
    q_value: Any,
    tolerance: float,
) -> pd.Series:
    if field not in cand.columns:
        return pd.Series(0.2, index=cand.index, dtype=float)

    if q_value is None or pd.isna(q_value):
        return cand[field].notna().astype(float) * 0.2

    q = float(q_value)
    vals = pd.to_numeric(cand[field], errors="coerce").astype(float)

    diff = (vals - q).abs() / max(abs(q), 1.0)
    sim = 1.0 - (diff / tolerance).clip(0, 1)
    sim = sim.fillna(0.2)

    return sim.astype(float)


def build_tfidf_matrix(names: pd.Series, max_features: int = 50000) -> Tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-zА-Яа-яЁё0-9]+\b",
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(names.fillna("").astype(str))
    LOGGER.info("TF-IDF matrix shape: %s", X.shape)

    return vectorizer, X


# ---------------------------------------------------------------------
# Выбор режима
# ---------------------------------------------------------------------

def select_mode_from_source_quality(
    source_quality: Dict[str, Any],
    min_evaluable_rate_for_validation: float = 0.60,
    min_pass_rate_evaluable_for_validation: float = 0.55,
    weak_evaluable_rate: float = 0.45,
    weak_pass_rate_evaluable: float = 0.90,
) -> Tuple[str, str]:
    use_true_analogs = (
        source_quality["evaluable_rate"] >= min_evaluable_rate_for_validation
        and source_quality["pass_rate_evaluable"] >= min_pass_rate_evaluable_for_validation
    )

    use_weak_analogs_source = (
        not use_true_analogs
        and source_quality["evaluable_rate"] >= weak_evaluable_rate
        and source_quality["pass_rate_evaluable"] >= weak_pass_rate_evaluable
    )

    if use_true_analogs:
        return (
            "A_true_analogs_validation",
            "Analogs можно использовать как полноценный эталон для диагностики и слабой разметки.",
        )

    if use_weak_analogs_source:
        return (
            "B_business_heuristics_with_weak_evaluable_source",
            (
                "Полный Analogs нельзя использовать как чистый эталон из-за низкой проверяемости, "
                "но среди технических пар его качество высокое. "
                "Можно использовать проверяемую часть как слабую разметку."
            ),
        )

    return (
        "B_business_heuristics",
        "Analogs слишком шумный или слишком непроверяемый, используется бизнес-скоринг.",
    )


# ---------------------------------------------------------------------
# Главный бизнес-рекомендатель
# ---------------------------------------------------------------------

DEFAULT_TARGET_CATEGORY_KEYWORDS: List[str] = [
    "сплит",
    "кондицион",
    "кондиционер",
]


class BusinessRecommender:

    def __init__(
        self,
        business_specs: Optional[List[Dict[str, Any]]] = None,
        max_candidates: int = 300,
        default_min_score: float = 0.0,
        target_category_keywords: Optional[List[str]] = None,
        analogs_validation_threshold: float = 0.60,
        enable_source_quality_validation: bool = True,
        tfidf_max_features: int = 50000,
        min_evaluable_rate_for_validation: float = 0.60,
        min_pass_rate_evaluable_for_validation: float = 0.55,
        weak_evaluable_rate: float = 0.45,
        weak_pass_rate_evaluable: float = 0.90,
    ):
        self.business_specs = business_specs or DEFAULT_BUSINESS_SPECS
        self.business_context = build_business_context(self.business_specs)
        self.max_candidates = int(max_candidates)
        self.default_min_score = float(default_min_score)

        self.target_category_keywords = target_category_keywords or DEFAULT_TARGET_CATEGORY_KEYWORDS
        self.analogs_validation_threshold = float(analogs_validation_threshold)
        self.enable_source_quality_validation = bool(enable_source_quality_validation)
        self.tfidf_max_features = int(tfidf_max_features)

        self.min_evaluable_rate_for_validation = float(min_evaluable_rate_for_validation)
        self.min_pass_rate_evaluable_for_validation = float(min_pass_rate_evaluable_for_validation)
        self.weak_evaluable_rate = float(weak_evaluable_rate)
        self.weak_pass_rate_evaluable = float(weak_pass_rate_evaluable)

        self.df: Optional[pd.DataFrame] = None
        self.df_indexed: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.X_text = None
        self.text_sim_matrix = None

        self.id_to_idx: Dict[str, int] = {}
        self.id_to_pos: Dict[str, int] = {}
        self.all_positions: List[int] = []

        self.selected_mode = "B_business_heuristics"
        self.mode_reason = (
            "Пока нет данных для выбора режима; "
            "по умолчанию используется бизнес-эвристика."
        )

        self.source_quality_metrics: Optional[Dict[str, Any]] = None
        self.evaluable_ground_truth: Optional[Dict[str, Set[str]]] = None

        self._initialized = False

    # -----------------------------------------------------------------
    # Инфраструктура
    # -----------------------------------------------------------------

    @property
    def catalog_size(self) -> int:
        if self.df is None:
            return 0
        return int(len(self.df))

    def get_business_weights(self) -> Dict[str, float]:
        return dict(self.business_context.weights)

    def get_tolerances(self) -> Dict[str, Optional[float]]:
        return dict(self.business_context.field_to_tolerance)

    def get_product_row(self, product_id: str) -> Optional[pd.Series]:
        """
        Возвращает строку товара по ID, если сервис инициализирован.
        """
        if self.df is None or self.df_indexed is None:
            return None

        product_id = str(product_id).strip()
        if product_id not in self.df_indexed.index:
            return None

        row = self.df_indexed.loc[product_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        return row

    def initialize(self, raw_records: Iterable[Dict[str, Any]]) -> "BusinessRecommender":

        raw_records = list(raw_records)

        profiles_df = build_profiles_from_records(raw_records)

        if profiles_df.empty:
            raise ValueError("Не удалось построить профили товаров из переданных записей")

        df = filter_by_name_keywords(profiles_df, self.target_category_keywords).copy()

        if df.empty:
            LOGGER.warning(
                "После фильтрации по ключевым словам %s осталось 0 товаров. "
                "Повторно пробую использовать весь каталог без фильтра.",
                self.target_category_keywords,
            )
            df = profiles_df.copy()

        core_numeric_fields = [
            "area_m2",
            "cooling_power_kw",
            "heating_power_kw",
            "noise_indoor_db",
            "min_operating_temp_c",
        ]

        binary_fields = [
            "is_inverter",
            "has_wifi_module",
            "has_wifi_app_control",
        ]

        categorical_fields = [
            "color_inner_body",
            "refrigerant",
            "energy_class",
        ]

        all_business_fields = core_numeric_fields + binary_fields + categorical_fields

        for field_name in core_numeric_fields:
            if field_name in df.columns:
                df[field_name] = pd.to_numeric(df[field_name], errors="coerce")

        for field_name in binary_fields:
            if field_name not in df.columns:
                df[field_name] = None

        for field_name in categorical_fields:
            if field_name not in df.columns:
                df[field_name] = None

        df = df.reset_index(drop=True)

        texts = df["normalized_name"].fillna("").astype(str).tolist()

        effective_min_df = 1 if len(texts) < 50 else 2

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=effective_min_df,
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[A-Za-zА-Яа-яЁё0-9]+\b",
            dtype=np.float32,
        )

        self.X_text = self.vectorizer.fit_transform(texts)

        try:
            self.text_sim_matrix = cosine_similarity(self.X_text, self.X_text).astype(np.float32)
        except Exception as exc:
            LOGGER.warning(
                "Не удалось построить dense TF-IDF similarity matrix: %s. "
                "Кандидаты будут строиться без text_sim.",
                exc,
            )
            self.text_sim_matrix = None

        self.df = df
        self.df_indexed = self.df.set_index("id", drop=False)

        self.id_to_idx = {
            str(pid): idx
            for idx, pid in enumerate(df["id"].astype(str).tolist())
        }

        self.id_to_pos = dict(self.id_to_idx)
        self.all_positions = list(df.index)

        if self.enable_source_quality_validation:
            ground_truth = build_analog_ground_truth(df)
            source_quality_result = compute_analogs_source_quality(
                df=df,
                ground_truth=ground_truth,
                context=self.business_context,
                threshold=self.analogs_validation_threshold,
            )

            self.source_quality_metrics = source_quality_result["source_quality_metrics"]
            self.evaluable_ground_truth = source_quality_result["evaluable_ground_truth"]

            self.selected_mode, self.mode_reason = select_mode_from_source_quality(
                source_quality=self.source_quality_metrics,
                min_evaluable_rate_for_validation=self.min_evaluable_rate_for_validation,
                min_pass_rate_evaluable_for_validation=self.min_pass_rate_evaluable_for_validation,
                weak_evaluable_rate=self.weak_evaluable_rate,
                weak_pass_rate_evaluable=self.weak_pass_rate_evaluable,
            )
        else:
            self.source_quality_metrics = None
            self.evaluable_ground_truth = None
            self.selected_mode = "B_business_heuristics"
            self.mode_reason = "Валидация источника Analogs отключена при инициализации."

        self._initialized = True

        LOGGER.info(
            "BusinessRecommender initialized: catalog=%d, mode=%s",
            self.catalog_size,
            self.selected_mode,
        )

        return self

    def _require_ready(self) -> None:
        if not self._initialized or self.df is None:
            raise RuntimeError("BusinessRecommender не инициализирован")

    # -----------------------------------------------------------------
    # Поиск запроса
    # -----------------------------------------------------------------

    def find_query_id(self, query: str) -> Optional[str]:

        self._require_ready()

        if not query:
            return None

        q_raw = str(query).strip()

        if q_raw in self.id_to_idx:
            return q_raw

        code_mask = self.df["code"].astype(str).str.strip() == q_raw
        if code_mask.any():
            return str(self.df.loc[code_mask, "id"].iloc[0])

        q_norm = normalize_text(q_raw)

        name_mask = self.df["normalized_name"].astype(str).str.strip() == q_norm
        if name_mask.any():
            return str(self.df.loc[name_mask, "id"].iloc[0])

        contains_mask = self.df["normalized_name"].astype(str).str.contains(
            re.escape(q_norm),
            na=False,
            regex=True,
        )

        if contains_mask.any():
            return str(self.df.loc[contains_mask, "id"].iloc[0])

        return None

    # -----------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------

    def get_candidate_positions(
        self,
        query_id: str,
        max_candidates: Optional[int] = None,
    ) -> List[int]:

        self._require_ready()

        query_id = str(query_id)

        if query_id not in self.id_to_idx:
            return []

        if max_candidates is None:
            max_candidates = self.max_candidates

        max_candidates = int(max_candidates)

        qidx = self.id_to_idx[query_id]
        qrow = self.df.iloc[qidx]

        pool = [pos for pos in self.all_positions if pos != qidx]

        if not pool:
            return []

        cand = self.df.loc[pool]

        score = pd.Series(0.0, index=cand.index, dtype=float)

        if self.text_sim_matrix is not None and query_id in self.id_to_pos:
            qpos = self.id_to_pos[query_id]
            try:
                text_sim = np.asarray(self.text_sim_matrix[qpos][pool], dtype=float)
                score += 0.45 * text_sim
            except Exception as exc:
                LOGGER.warning("Не удалось использовать text_sim для candidates: %s", exc)

        score += 0.25 * _analog_numeric_similarity_series(
            cand=cand,
            field="cooling_power_kw",
            q_value=qrow.get("cooling_power_kw"),
            tolerance=0.35,
        )

        score += 0.15 * _analog_numeric_similarity_series(
            cand=cand,
            field="area_m2",
            q_value=qrow.get("area_m2"),
            tolerance=0.35,
        )

        score += 0.10 * _analog_numeric_similarity_series(
            cand=cand,
            field="heating_power_kw",
            q_value=qrow.get("heating_power_kw"),
            tolerance=0.35,
        )

        score += 0.02 * _analog_numeric_similarity_series(
            cand=cand,
            field="noise_indoor_db",
            q_value=qrow.get("noise_indoor_db"),
            tolerance=0.50,
        )

        q_inverter = qrow.get("is_inverter")

        if pd.notna(q_inverter):
            inv_match = (cand["is_inverter"] == q_inverter).astype(float)
            score += 0.02 * inv_match
        else:
            score += 0.02 * cand["is_inverter"].notna().astype(float) * 0.5

        q_ref = qrow.get("refrigerant")

        if q_ref is not None and not pd.isna(q_ref):
            ref_match = (
                cand["refrigerant"]
                .astype(str)
                .str.strip()
                .str.lower()
                == str(q_ref).strip().lower()
            ).astype(float)
            score += 0.01 * ref_match

        ordered = score.sort_values(ascending=False, kind="mergesort").index.tolist()

        return [int(pos) for pos in ordered[:max_candidates]]

    # -----------------------------------------------------------------
    # Основной подбор аналогов
    # -----------------------------------------------------------------

    def get_analogs_by_id(
        self,
        product_id: str,
        k: int = 5,
        min_score: Optional[float] = None,
        include_details: bool = True,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:

        self._require_ready()

        product_id = str(product_id)

        if product_id not in self.id_to_idx:
            return {
                "query_id": product_id,
                "query_code": None,
                "query_name": None,
                "results": [],
                "total_candidates_generated": 0,
                "total_candidates_validated": 0,
                "total_candidates_passed": 0,
            }

        if min_score is None:
            min_score = self.default_min_score

        min_score = float(min_score)

        q_idx = self.id_to_idx[product_id]
        query_row = self.df.iloc[q_idx]

        candidate_positions = self.get_candidate_positions(
            query_id=product_id,
            max_candidates=max_candidates,
        )

        results: List[Dict[str, Any]] = []
        total_validated = 0

        for cpos in candidate_positions:
            candidate_row = self.df.iloc[cpos]

            scoring = business_pair_score(
                query_row=query_row,
                candidate_row=candidate_row,
                context=self.business_context,
            )

            total_validated += 1

            valid_field_count = int(scoring.get("valid_field_count", 0))
            core_valid_field_count = int(scoring.get("core_valid_field_count", 0))

            if valid_field_count == 0:
                continue

            if core_valid_field_count == 0:
                continue

            if scoring["hard_fail"]:
                continue

            text_sim = 0.0
            if self.text_sim_matrix is not None and product_id in self.id_to_pos:
                try:
                    text_sim = float(self.text_sim_matrix[self.id_to_pos[product_id], int(cpos)])
                except Exception:
                    text_sim = 0.0

            item = {
                "id": str(candidate_row["id"]),
                "code": candidate_row.get("code"),
                "name": candidate_row.get("name"),
                "score": float(scoring["score"]),
                "valid_score": scoring.get("valid_score"),
                "confidence": float(scoring.get("confidence", 0.0)),
                "status": scoring.get("status"),
                "hard_fail": bool(scoring["hard_fail"]),
                "text_similarity": text_sim,
                "valid_field_count": valid_field_count,
                "core_valid_field_count": core_valid_field_count,
                "valid_fields": scoring.get("valid_fields", []),
                "missing_fields": scoring.get("missing_fields", []),
                "hard_fail_fields": scoring.get("hard_fail_fields", []),
                "reasons": scoring.get("reasons", []),
                "area_m2": candidate_row.get("area_m2"),
                "cooling_power_kw": candidate_row.get("cooling_power_kw"),
                "heating_power_kw": candidate_row.get("heating_power_kw"),
                "noise_indoor_db": candidate_row.get("noise_indoor_db"),
                "min_operating_temp_c": candidate_row.get("min_operating_temp_c"),
                "is_inverter": candidate_row.get("is_inverter"),
                "has_wifi_module": candidate_row.get("has_wifi_module"),
                "has_wifi_app_control": candidate_row.get("has_wifi_app_control"),
                "color_inner_body": candidate_row.get("color_inner_body"),
                "refrigerant": candidate_row.get("refrigerant"),
                "energy_class": candidate_row.get("energy_class"),
            }

            if include_details:
                item["component_details"] = scoring.get("component_details", {})

            results.append(item)

        results = sorted(
            results,
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("text_similarity", 0.0)),
            ),
            reverse=True,
        )

        if min_score > 0.0:
            results = [
                row for row in results
                if float(row.get("score", 0.0)) >= min_score
            ]

        k = int(k)
        results = results[:k]

        return {
            "query_id": product_id,
            "query_code": query_row.get("code"),
            "query_name": query_row.get("name"),
            "results": results,
            "total_candidates_generated": len(candidate_positions),
            "total_candidates_validated": int(total_validated),
            "total_candidates_passed": len(results),
        }

    def get_analogs_by_query(
        self,
        query: str,
        k: int = 5,
        min_score: Optional[float] = None,
        include_details: bool = True,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._require_ready()

        query_id = self.find_query_id(query)

        if query_id is None:
            return {
                "query_id": None,
                "query_code": None,
                "query_name": None,
                "results": [],
                "total_candidates_generated": 0,
                "total_candidates_validated": 0,
                "total_candidates_passed": 0,
            }

        return self.get_analogs_by_id(
            product_id=query_id,
            k=k,
            min_score=min_score,
            include_details=include_details,
            max_candidates=max_candidates,
        )

    # -----------------------------------------------------------------
    # Статистика и диагностика
    # -----------------------------------------------------------------

    def _valid_coverage(self) -> Dict[str, float]:

        self._require_ready()

        coverage: Dict[str, float] = {}

        for spec in self.business_specs:
            field = spec["Field"]

            if field not in self.df.columns:
                coverage[field] = 0.0
                continue

            values = self.df[field]

            def is_valid_value(value: Any) -> bool:
                if value is None:
                    return False

                if isinstance(value, bool):
                    return True

                if isinstance(value, float):
                    return not math.isnan(value)

                if isinstance(value, str):
                    return not is_nullish_value(value)

                return True

            coverage[field] = float(values.map(is_valid_value).mean())

        return coverage

    def _analogs_diagnostics(self) -> Dict[str, Any]:

        if self.df is None or self.df.empty:
            return {}

        valid_ids = set(self.df["id"].astype(str).tolist())

        def has_any_analogs(value: Any) -> bool:
            return isinstance(value, list) and len(value) > 0

        def has_valid_analog(value: Any) -> bool:
            if not isinstance(value, list):
                return False
            return any(str(item).strip() in valid_ids for item in value)

        products_with_any_analogs = int(self.df["analogs"].map(has_any_analogs).sum())
        products_with_valid_analogs = int(self.df["analogs"].map(has_valid_analog).sum())

        total_refs = 0
        unique_targets: Set[str] = set()

        for analog_list in self.df["analogs"]:
            if isinstance(analog_list, list):
                total_refs += len(analog_list)
                unique_targets.update(
                    str(item).strip()
                    for item in analog_list
                    if str(item).strip()
                )

        merge_conflict_rows = 0
        if "_merge_conflicts" in self.df.columns:
            merge_conflict_rows = int(
                self.df["_merge_conflicts"].map(
                    lambda x: isinstance(x, list) and len(x) > 0
                ).sum()
            )

        return {
            "products_with_any_analogs": products_with_any_analogs,
            "products_with_valid_analogs": products_with_valid_analogs,
            "total_analog_references": total_refs,
            "unique_analog_targets": len(unique_targets),
            "products_with_merge_conflicts": merge_conflict_rows,
        }

    def get_stats(self) -> Dict[str, Any]:

        if not self._initialized or self.df is None:
            return {
                "initialized": False,
                "catalog_size": 0,
                "selected_mode": "not_initialized",
                "mode_reason": "Система не инициализирована",
                "business_weights": {},
                "valid_coverage": {},
                "source_quality": None,
                "analogs_pass_rate": None,
                "analogs_diagnostics": {},
            }

        source_quality = self.source_quality_metrics or {}

        return {
            "initialized": True,
            "catalog_size": self.catalog_size,
            "selected_mode": self.selected_mode,
            "mode_reason": self.mode_reason,
            "business_weights": self.get_business_weights(),
            "valid_coverage": self._valid_coverage(),
            "source_quality": source_quality,
            "analogs_pass_rate": source_quality.get("pass_rate_all"),
            "analogs_diagnostics": self._analogs_diagnostics(),
        }
