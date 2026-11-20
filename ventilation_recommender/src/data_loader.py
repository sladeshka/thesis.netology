"""
Загрузка и подготовка данных каталога.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from .config import CATALOG_FILE, DATA_DIR

class DataLoader:
    """Загрузчик данных каталога."""

    @staticmethod
    def load_raw_json(filepath: Path) -> List[Dict]:
        """Загружает JSON-файл и возвращает список товаров."""
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("products", "items", "data", "catalog"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        raise ValueError("Неизвестный формат JSON-данных")

    @staticmethod
    def load_catalog(filepath: Optional[Path] = None) -> pd.DataFrame:
        """Загружает каталог в DataFrame."""
        if filepath is None:
            filepath = DATA_DIR / CATALOG_FILE
        raw_items = DataLoader.load_raw_json(filepath)
        normalized_items = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            product_id = ""
            if isinstance(item.get("_id"), dict):
                product_id = str(item["_id"].get("$uuid", ""))
            elif item.get("_id") is not None:
                product_id = str(item["_id"])
            elif item.get("id") is not None:
                product_id = str(item["id"])
            normalized_items.append(
                {
                    "id": product_id,
                    "Name": item.get("Name", item.get("name", "")),
                    "Code": item.get("Code", item.get("code", "")),
                    "Category": item.get("Category", item.get("category", "")),
                    "Characteristics": item.get("Characteristics", item.get("characteristics", [])),
                    "Analogs": item.get("Analogs", item.get("analogs", [])),
                }
            )
        return pd.DataFrame(normalized_items)

    @staticmethod
    def filter_by_category(
        df: pd.DataFrame,
        keywords: List[str],
        exclude_patterns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Фильтрует DataFrame по категории или названию."""
        if df.empty:
            return df.copy()
        category_field = None
        for field in ("Category", "category", "CategoryName", "category_name"):
            if field in df.columns:
                category_field = field
                break
        if category_field is not None:
            filtered_df = DataLoader._filter_by_field(df, category_field, keywords, exclude_patterns)
            if not filtered_df.empty:
                return filtered_df
        name_field = "Name" if "Name" in df.columns else "name"
        return DataLoader._filter_by_name_keywords(df, name_field, keywords, exclude_patterns)

    @staticmethod
    def _filter_by_field(
        df: pd.DataFrame,
        field: str,
        keywords: List[str],
        exclude_patterns: Optional[List[str]],
    ) -> pd.DataFrame:
        values = df[field].fillna("").astype(str).str.lower()
        mask = pd.Series(False, index=df.index)
        for keyword in keywords:
            mask |= values.str.contains(keyword.lower(), regex=False)
        if exclude_patterns:
            for pattern in exclude_patterns:
                mask &= ~values.str.contains(pattern.lower(), regex=False)
        return df[mask].copy()

    @staticmethod
    def _filter_by_name_keywords(
        df: pd.DataFrame,
        name_field: str,
        keywords: List[str],
        exclude_patterns: Optional[List[str]],
    ) -> pd.DataFrame:
        values = df[name_field].fillna("").astype(str).str.lower()
        mask = pd.Series(False, index=df.index)
        for keyword in keywords:
            mask |= values.str.contains(keyword.lower(), regex=False)
        if exclude_patterns:
            for pattern in exclude_patterns:
                mask &= ~values.str.contains(pattern.lower(), regex=False)
        return df[mask].copy()

    @staticmethod
    def load_analogs_mapping(filepath: Path) -> Dict[str, List[str]]:
        """Загружает маппинг известных аналогов для валидации."""
        items = DataLoader.load_raw_json(filepath)
        mapping: Dict[str, List[str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            product_id = str(item.get("id", ""))
            if not product_id and isinstance(item.get("_id"), dict):
                product_id = str(item["_id"].get("$uuid", ""))
            analogs = item.get("Analogs", item.get("analogs", []))
            if product_id and isinstance(analogs, list):
                mapping[product_id] = [str(analog) for analog in analogs]
        return mapping

    @staticmethod
    def explore_data(df: pd.DataFrame) -> Dict:
        """Разведочный анализ данных."""
        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_values": {},
        }
        for col in ("Name", "Code", "id"):
            if col in df.columns:
                stats["unique_values"][col] = int(df[col].nunique())
        return stats

    @staticmethod
    def extract_characteristics_sample(df: pd.DataFrame, n_samples: int = 5) -> List[Dict]:
        """Извлекает примеры характеристик."""
        samples = []
        for idx in range(min(n_samples, len(df))):
            record = df.iloc[idx]
            sample = {
                "name": record.get("Name", "N/A"),
                "characteristics_count": 0,
                "numeric_features": {},
                "categorical_features": {},
            }
            chars = record.get("Characteristics", [])
            if isinstance(chars, list):
                sample["characteristics_count"] = len(chars)
                for char in chars[:20]:
                    if not isinstance(char, dict):
                        continue
                    value = char.get("Value")
                    if value is None:
                        continue
                    name = char.get("Name", "")
                    text_value = str(value)
                    try:
                        float(text_value.replace(",", ".").replace(" ", ""))
                        sample["numeric_features"][name] = text_value
                    except ValueError:
                        if len(text_value) < 50:
                            sample["categorical_features"][name] = text_value
            samples.append(sample)
        return samples


def load_and_prepare_catalog(filepath: Path, category_keywords: List[str]) -> pd.DataFrame:
    """Загружает и фильтрует каталог."""
    df = DataLoader.load_catalog(filepath)
    return DataLoader.filter_by_category(df, keywords=category_keywords, exclude_patterns=["архивн", "удален"])
