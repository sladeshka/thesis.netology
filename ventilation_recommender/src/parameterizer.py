"""
Параметризация товаров и построение матрицы признаков.
"""

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

@dataclass
class ProductFeatures:
    """Вектор признаков одного товара."""
    product_id: str
    name: str
    numeric_values: np.ndarray
    categorical_values: np.ndarray
    categorical_raw: Dict[str, Optional[str]]
    has_all_numeric: bool
    missing_features: List[str]

@dataclass
class FeatureMatrix:
    """Матрица признаков каталога."""
    product_ids: np.ndarray
    names: np.ndarray
    numeric_matrix: np.ndarray
    categorical_matrix: np.ndarray
    numeric_scaler: Optional[MinMaxScaler]
    category_encoders: Dict[str, LabelEncoder]
    feature_names_numeric: List[str]
    feature_names_categorical: List[str]

class Parameterizer:
    """Параметризатор товаров."""

    def __init__(self):
        self.numeric_feature_names = [f.name for f in NUMERIC_FEATURES]
        self.categorical_feature_names = [f.name for f in CATEGORICAL_FEATURES]
        self.default_values = {
            f.name: f.default_value
            for f in NUMERIC_FEATURES
            if f.default_value is not None
        }
        self._is_fitted = False
        self.feature_matrix: Optional[FeatureMatrix] = None

    def product_to_features(self, product: Dict, for_catalog: bool = True) -> ProductFeatures:
        """Преобразует товар в вектор признаков."""
        product_id = str(product.get("id", product.get("_id", "")))
        if not product_id and isinstance(product.get("_id"), dict):
            product_id = str(product["_id"].get("$uuid", ""))
        name = str(product.get("name", product.get("Name", "")))
        if "characteristics" in product and isinstance(product["characteristics"], dict):
            chars = product["characteristics"]
        elif "Characteristics" in product and isinstance(product["Characteristics"], dict):
            chars = product["Characteristics"]
        elif for_catalog:
            chars = self._parse_characteristics_list(product.get("Characteristics", []))
        else:
            chars = {}
        numeric_values: List[float] = []
        missing_features: List[str] = []
        for feature_name in self.numeric_feature_names:
            raw_value = chars.get(feature_name)
            value: Optional[float] = None
            if raw_value is not None and str(raw_value).strip():
                try:
                    value = float(str(raw_value).replace(",", "."))
                except (ValueError, AttributeError):
                    value = None
            if value is None:
                missing_features.append(feature_name)
                default_value = self.default_values.get(feature_name)
                value = default_value if default_value is not None else 0.0
            numeric_values.append(value)
        categorical_raw: Dict[str, Optional[str]] = {}
        for feature_name in self.categorical_feature_names:
            raw_value = chars.get(feature_name)
            if raw_value is not None and str(raw_value).strip():
                categorical_raw[feature_name] = str(raw_value).strip()
            else:
                categorical_raw[feature_name] = None
        return ProductFeatures(
            product_id=product_id,
            name=name,
            numeric_values=np.array(numeric_values, dtype=float),
            categorical_values=np.zeros(len(self.categorical_feature_names), dtype=float),
            categorical_raw=categorical_raw,
            has_all_numeric=not missing_features,
            missing_features=missing_features,
        )

    def build_catalog_matrix(self, catalog: List[Dict]) -> FeatureMatrix:
        """Строит матрицу признаков для каталога."""
        if not catalog:
            return FeatureMatrix(
                product_ids=np.array([], dtype=object),
                names=np.array([], dtype=object),
                numeric_matrix=np.zeros((0, len(self.numeric_feature_names)), dtype=float),
                categorical_matrix=np.zeros((0, len(self.categorical_feature_names)), dtype=float),
                numeric_scaler=None,
                category_encoders={},
                feature_names_numeric=self.numeric_feature_names,
                feature_names_categorical=[f.display_name for f in CATEGORICAL_FEATURES],
            )

        n_products = len(catalog)
        n_numeric = len(self.numeric_feature_names)
        n_categorical = len(self.categorical_feature_names)
        product_ids: List[str] = []
        names: List[str] = []
        numeric_matrix = np.zeros((n_products, n_numeric), dtype=float)
        categorical_raw_data: Dict[str, List[object]] = {name: [] for name in self.categorical_feature_names}
        for i, product in enumerate(catalog):
            features = self.product_to_features(product, for_catalog=True)
            product_ids.append(features.product_id)
            names.append(features.name)
            numeric_matrix[i] = np.nan_to_num(features.numeric_values, nan=0.0)
            for cat_name in self.categorical_feature_names:
                value = features.categorical_raw.get(cat_name)
                if cat_name in {"Инверторная технология", "Wi-Fi модуль"}:
                    categorical_raw_data[cat_name].append(self._normalize_bool(value))
                else:
                    categorical_raw_data[cat_name].append(value or "__MISSING__")
        category_encoders: Dict[str, LabelEncoder] = {}
        categorical_matrix = np.zeros((n_products, n_categorical), dtype=float)
        for j, cat_name in enumerate(self.categorical_feature_names):
            values = categorical_raw_data[cat_name]
            if cat_name in {"Инверторная технология", "Wi-Fi модуль"}:
                categorical_matrix[:, j] = np.array([1.0 if value == 1 else 0.0 for value in values], dtype=float)
                continue
            string_values = [str(value) for value in values]
            unique_values = sorted(set(string_values + ["__UNKNOWN__"]))
            encoder = LabelEncoder()
            encoder.fit(unique_values)
            category_encoders[cat_name] = encoder
            encoded = np.array(
                [
                    encoder.transform([value if value in encoder.classes_ else "__UNKNOWN__"])[0]
                    for value in string_values
                ],
                dtype=float,
            )
            max_value = max(len(unique_values) - 1, 1)
            categorical_matrix[:, j] = encoded / max_value
        numeric_matrix = np.nan_to_num(numeric_matrix, nan=0.0)
        scaler = MinMaxScaler()
        numeric_normalized = scaler.fit_transform(numeric_matrix)
        self.feature_matrix = FeatureMatrix(
            product_ids=np.array(product_ids, dtype=object),
            names=np.array(names, dtype=object),
            numeric_matrix=numeric_normalized,
            categorical_matrix=categorical_matrix,
            numeric_scaler=scaler,
            category_encoders=category_encoders,
            feature_names_numeric=self.numeric_feature_names,
            feature_names_categorical=[f.display_name for f in CATEGORICAL_FEATURES],
        )
        self._is_fitted = True
        return self.feature_matrix

    def text_to_features(self, text: str, preprocessor) -> Tuple[np.ndarray, Dict[str, str]]:
        """Извлекает признаки из текстового запроса."""
        parsed = preprocessor.process(text)
        numeric_values = self._extract_numeric_from_text(parsed.tokens)
        categorical_values: Dict[str, str] = {}
        if parsed.brand:
            categorical_values["Бренд"] = parsed.brand
        return np.array(numeric_values, dtype=float), categorical_values

    def _parse_characteristics_list(self, chars_list: List[Dict]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for char in chars_list:
            if not isinstance(char, dict):
                continue
            name = str(char.get("Name", "")).strip()
            value = char.get("Value")
            if name and value is not None:
                if isinstance(value, str) and not value.strip():
                    continue
                result[name] = value
        return result

    def _normalize_bool(self, value: Optional[str]) -> int:
        if not value:
            return -1
        value_lower = str(value).lower().strip()
        positive_prefixes = ("да", "ес", "tru", "ye", "+", "1")
        if any(value_lower.startswith(prefix) for prefix in positive_prefixes):
            return 1
        if "нет" in value_lower or value_lower in {"false", "0", "-"}:
            return 0
        return -1

    def _extract_numeric_from_text(self, tokens: List[str]) -> List[float]:
        values = [0.0 for _ in self.numeric_feature_names]
        for token in tokens:
            try:
                numeric_token = float(token.replace(",", "."))
            except ValueError:
                continue
            integer_value = int(numeric_token) if numeric_token.is_integer() else numeric_token
            if 7 <= integer_value <= 200 and values[0] == 0:
                values[0] = float(integer_value)
            if 5000 <= integer_value <= 48000 and values[1] == 0:
                values[1] = float(integer_value) * 0.000293071
            if 15 <= integer_value <= 60 and values[3] == 0:
                values[3] = float(integer_value)
        return values

    def get_default_matrix_row(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает усреднённый вектор признаков."""
        if self._is_fitted and self.feature_matrix is not None and len(self.feature_matrix.numeric_matrix) > 0:
            numeric_mean = self.feature_matrix.numeric_matrix.mean(axis=0)
        else:
            numeric_mean = np.full(len(self.numeric_feature_names), 0.5, dtype=float)
        categorical_default = np.zeros(len(self.categorical_feature_names), dtype=float)
        return numeric_mean, categorical_default


def save_feature_matrix(feature_matrix: FeatureMatrix, filepath: str) -> None:
    """Сохраняет матрицу признаков."""
    save_data = {
        "product_ids": feature_matrix.product_ids,
        "names": feature_matrix.names,
        "numeric_matrix": feature_matrix.numeric_matrix,
        "categorical_matrix": feature_matrix.categorical_matrix,
        "feature_names_numeric": feature_matrix.feature_names_numeric,
        "feature_names_categorical": feature_matrix.feature_names_categorical,
    }
    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)


def load_feature_matrix(filepath: str) -> FeatureMatrix:
    """Загружает матрицу признаков."""
    with open(filepath, "rb") as f:
        save_data = pickle.load(f)
    return FeatureMatrix(
        product_ids=save_data["product_ids"],
        names=save_data["names"],
        numeric_matrix=save_data["numeric_matrix"],
        categorical_matrix=save_data["categorical_matrix"],
        numeric_scaler=None,
        category_encoders={},
        feature_names_numeric=save_data["feature_names_numeric"],
        feature_names_categorical=save_data["feature_names_categorical"],
    )
