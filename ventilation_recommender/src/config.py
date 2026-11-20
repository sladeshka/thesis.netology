"""
Конфигурация проекта. "Система подбора аналогов вентиляционного оборудования на базе параметризатора и рекомендательных систем для интернет-магазина"
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

for directory in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CATALOG_FILE = "catalog_products.json"

TARGET_CATEGORY_KEYWORDS = [
    "сплит",
    "кондицион",
    "кондиционер",
]

DEFAULT_K = 10

DEFAULT_MIN_SCORE = 0.0

MAX_CANDIDATES_PER_QUERY = 300

ANALOGS_VALIDATION_THRESHOLD = 0.60

TFIDF_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
