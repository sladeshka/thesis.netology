"""
REST API для системы подбора аналогов вентиляционного и климатического оборудования.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .business_service import (
    DEFAULT_BUSINESS_SPECS,
    DEFAULT_TARGET_CATEGORY_KEYWORDS,
    BusinessRecommender,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger("analog_api")

API_VERSION = os.getenv("ANALOG_API_VERSION", "3.2.0")
PROJECT_NAME = "ventilation_analog_selection"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = os.getenv("ANALOG_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("ANALOG_API_PORT", "8000"))
    debug: bool = os.getenv("ANALOG_API_DEBUG", "false").lower() in {"1", "true", "yes"}

    data_path: Optional[str] = os.getenv(
        "ANALOG_DATA_PATH",
        "data/raw/catalog_products.json",
    )

    max_records: Optional[int] = None

    default_k: int = int(os.getenv("ANALOG_DEFAULT_K", "10"))
    default_min_score: float = float(os.getenv("ANALOG_DEFAULT_MIN_SCORE", "0.0"))

    max_candidates_per_query: int = int(os.getenv("ANALOG_MAX_CANDIDATES", "300"))
    tfidf_max_features: int = int(os.getenv("ANALOG_TFIDF_MAX_FEATURES", "50000"))

    allow_demo_catalog: bool = os.getenv(
        "ANALOG_ALLOW_DEMO_CATALOG",
        "false",
    ).lower() in {"1", "true", "yes"}

    target_category_keywords: List[str] = [
        "сплит",
        "кондицион",
        "кондиционер",
    ]


settings = Settings()


class PredictRequest(BaseModel):

    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=settings.default_k, ge=1, le=50)
    min_score: float = Field(default=settings.default_min_score, ge=0.0, le=1.0)
    include_details: bool = Field(
        default=True,
        description="Возвращать component_details и reasons.",
    )


class PredictByIdRequest(BaseModel):

    product_id: str = Field(..., min_length=1)
    k: int = Field(default=settings.default_k, ge=1, le=50)
    min_score: float = Field(default=settings.default_min_score, ge=0.0, le=1.0)
    include_details: bool = Field(default=True)


class ComponentDetail(BaseModel):

    score: float = Field(ge=0.0, le=1.0)
    component_score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    hard_fail: bool = False
    skipped: bool = False
    missing: bool = False
    importance: Optional[int] = None
    kind: Optional[str] = None
    display_name: Optional[str] = None
    reason: Optional[str] = None


class AnalogItem(BaseModel):

    id: str
    code: Optional[str] = None
    name: str
    score: float = Field(ge=0.0, le=1.0)
    valid_score: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: Optional[str] = None
    hard_fail: bool = False

    text_similarity: float = Field(ge=0.0, le=1.0)

    valid_field_count: int = 0
    core_valid_field_count: int = 0
    valid_fields: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    hard_fail_fields: List[str] = Field(default_factory=list)
    reasons: Optional[List[str]] = None

    is_inverter: Optional[bool] = None
    area_m2: Optional[float] = None
    cooling_power_kw: Optional[float] = None
    heating_power_kw: Optional[float] = None
    noise_indoor_db: Optional[float] = None
    min_operating_temp_c: Optional[float] = None
    color_inner_body: Optional[str] = None
    has_wifi_module: Optional[bool] = None
    has_wifi_app_control: Optional[bool] = None
    refrigerant: Optional[str] = None
    energy_class: Optional[str] = None

    component_details: Optional[Dict[str, ComponentDetail]] = None


class PredictResponse(BaseModel):

    query: str
    query_id: Optional[str] = None
    query_type: str = "not_found"

    k: int
    results: List[AnalogItem]

    total_candidates_generated: int = 0
    total_candidates_validated: int = 0
    total_candidates_passed: int = 0

    catalog_size: int
    execution_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):

    status: str
    ready: bool
    version: str
    project: str
    selected_mode: Optional[str] = None
    catalog_size: int = 0
    message: Optional[str] = None


class BusinessParametersResponse(BaseModel):

    version: str
    selected_mode: str
    business_weights: Dict[str, float]
    tolerances_percent: Dict[str, Optional[float]]
    business_characteristics: List[Dict[str, object]]


class SourceQualityResponse(BaseModel):

    version: str
    selected_mode: str
    mode_reason: Optional[str] = None
    source_quality: Dict[str, Any]


class StatsResponse(BaseModel):

    initialized: bool
    catalog_size: int
    selected_mode: str
    mode_reason: Optional[str] = None

    business_weights: Dict[str, float]
    valid_coverage: Dict[str, float]

    source_quality: Optional[Dict[str, Any]] = None
    analogs_pass_rate: Optional[float] = None
    analogs_diagnostics: Dict[str, Any] = Field(default_factory=dict)


recommender: Optional[BusinessRecommender] = None
system_ready: bool = False


def load_raw_records(path: Path) -> List[Dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(f"Каталог не найден: {path}")

    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()
        f.seek(0)

        if first_char == "[":
            raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("Корень JSON должен быть списком записей")
            records = raw
        else:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Не удалось разобрать строку %d: %s", line_no, exc)

    return records


def generate_demo_records() -> List[Dict[str, Any]]:

    demo_products = []

    brands = ["Mitsubishi", "Daichi", "Ballu", "Electrolux", "Loriot", "Hisense"]
    series = ["SRK", "Bravo", "BSW", "Slide", "Sky", "Goal"]
    sizes = [7, 9, 12, 18, 24]

    for i, size in enumerate(sizes):
        for brand_idx, brand in enumerate(brands):
            cooling_kw = round(2.0 + size * 0.35 + brand_idx * 0.05, 2)
            heating_kw = round(cooling_kw * 1.25, 2)
            area = int(cooling_kw * 9)
            noise = 19 + (i + brand_idx) % 12

            demo_products.append(
                {
                    "Id": f"demo_{i}_{brand_idx}_{size}",
                    "Name": f"Инверторная сплит-система {brand} {series[brand_idx]}{size:02d}N8",
                    "Code": f"DEMO-{1000000 + i}_{brand_idx}_{size}",
                    "Characteristics": [
                        {"Name": "Инверторная технология", "Value": "Да"},
                        {"Name": "Эффективен для помещ. площадью до", "Value": str(area)},
                        {"Name": "Номинальная производительность охлаждения", "Value": str(cooling_kw)},
                        {"Name": "Номинальная производительность обогрева", "Value": str(heating_kw)},
                        {"Name": "Уровень шума внутр. блока", "Value": str(noise)},
                        {"Name": "Мин. рабочая температура воздуха для внешнего блока", "Value": "-15"},
                        {"Name": "Цвет корпуса внутр. блока", "Value": "Белый"},
                        {"Name": "Wi-Fi модуль", "Value": "Да" if (i + brand_idx) % 2 == 0 else "Нет"},
                        {
                            "Name": "Управление c мобильного приложения по Wi-Fi",
                            "Value": "Да" if (i + brand_idx) % 3 == 0 else "Нет",
                        },
                        {"Name": "Хладагент", "Value": "R32"},
                        {"Name": "Класс энергоэффективности", "Value": "A"},
                    ],
                    "Analogs": [],
                }
            )

    return demo_products


async def initialize_service() -> BusinessRecommender:

    service = BusinessRecommender(
        business_specs=DEFAULT_BUSINESS_SPECS,
        max_candidates=settings.max_candidates_per_query,
        default_min_score=settings.default_min_score,
        target_category_keywords=settings.target_category_keywords,
        tfidf_max_features=settings.tfidf_max_features,
    )

    if settings.data_path:
        path = Path(settings.data_path)
        if path.exists():
            LOGGER.info("Загрузка каталога из файла: %s", path)

            raw_records = load_raw_records(path)

            if settings.max_records is not None:
                raw_records = raw_records[: settings.max_records]

            if raw_records:
                import asyncio

                await asyncio.to_thread(service.initialize, raw_records)
                return service

    LOGGER.warning("Основной каталог не загружен")

    if settings.allow_demo_catalog:
        LOGGER.warning("Используется демо-каталог. Для production задай ANALOG_DATA_PATH")

        import asyncio

        demo_records = generate_demo_records()
        await asyncio.to_thread(service.initialize, demo_records)

        return service

    raise RuntimeError(
        "Каталог не найден и демо-режим запрещён. "
        "Установи ANALOG_DATA_PATH или ANALOG_ALLOW_DEMO_CATALOG=true."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):

    global recommender, system_ready

    LOGGER.info("Запуск API системы подбора аналогов v%s", API_VERSION)

    try:
        recommender = await initialize_service()
        system_ready = True
        LOGGER.info("Система готова. Каталог: %d", recommender.catalog_size)
    except Exception:
        LOGGER.exception("Ошибка инициализации системы")
        system_ready = False
        raise

    yield

    LOGGER.info("Завершение работы API")


app = FastAPI(
    title="Ventilation Equipment Analog Finder API",
    description=(
        "REST API для подбора аналогов климатического и вентиляционного оборудования "
        "на основе нормализованных технических характеристик и бизнес-правил."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> None:

    if not system_ready or recommender is None:
        raise HTTPException(status_code=503, detail="Система не готова к работе")


def _to_analog_item(result: Dict[str, Any], include_details: bool) -> AnalogItem:

    component_details = None

    if include_details and result.get("component_details"):
        component_details = {}

        for field_name, component in result["component_details"].items():
            component_score = float(component.get("component_score", component.get("score", 0.0)))

            component_details[field_name] = ComponentDetail(
                score=component_score,
                component_score=component_score,
                weight=float(component.get("weight", 0.0)),
                hard_fail=bool(component.get("hard_fail", False)),
                skipped=bool(component.get("skipped", False)),
                missing=bool(component.get("missing", False)),
                importance=component.get("importance"),
                kind=component.get("kind"),
                display_name=component.get("display_name"),
                reason=component.get("reason"),
            )

    return AnalogItem(
        id=str(result.get("id", "")),
        code=result.get("code"),
        name=str(result.get("name", "")),
        score=float(result.get("score", 0.0)),
        valid_score=result.get("valid_score"),
        confidence=float(result.get("confidence", 0.0)),
        status=result.get("status"),
        hard_fail=bool(result.get("hard_fail", False)),
        text_similarity=float(result.get("text_similarity", 0.0)),
        valid_field_count=int(result.get("valid_field_count", 0)),
        core_valid_field_count=int(result.get("core_valid_field_count", 0)),
        valid_fields=result.get("valid_fields", []) or [],
        missing_fields=result.get("missing_fields", []) or [],
        hard_fail_fields=result.get("hard_fail_fields", []) or [],
        reasons=result.get("reasons", []) if include_details else None,
        is_inverter=result.get("is_inverter"),
        area_m2=result.get("area_m2"),
        cooling_power_kw=result.get("cooling_power_kw"),
        heating_power_kw=result.get("heating_power_kw"),
        noise_indoor_db=result.get("noise_indoor_db"),
        min_operating_temp_c=result.get("min_operating_temp_c"),
        color_inner_body=result.get("color_inner_body"),
        has_wifi_module=result.get("has_wifi_module"),
        has_wifi_app_control=result.get("has_wifi_app_control"),
        refrigerant=result.get("refrigerant"),
        energy_class=result.get("energy_class"),
        component_details=component_details,
    )


def _detect_query_type(
    request_query: str,
    query_id: str,
    service_result: Dict[str, Any],
) -> str:

    query_clean = str(request_query).strip()

    if query_clean == str(query_id).strip():
        return "id"

    query_code = str(service_result.get("query_code") or "").strip()

    if query_code and query_clean == query_code:
        return "code"

    return "text"


@app.get("/health", response_model=HealthResponse)
async def health_check():

    if system_ready and recommender is not None:
        return HealthResponse(
            status="ok",
            ready=True,
            version=API_VERSION,
            project=PROJECT_NAME,
            selected_mode=recommender.selected_mode,
            catalog_size=recommender.catalog_size,
            message="Система готова к подбору аналогов",
        )

    return HealthResponse(
        status="error",
        ready=False,
        version=API_VERSION,
        project=PROJECT_NAME,
        catalog_size=0,
        message="Система не инициализирована",
    )


@app.get("/business-parameters", response_model=BusinessParametersResponse)
async def get_business_parameters():

    _require_ready()

    return BusinessParametersResponse(
        version=API_VERSION,
        selected_mode=recommender.selected_mode,
        business_weights=recommender.get_business_weights(),
        tolerances_percent=recommender.get_tolerances(),
        business_characteristics=[
            {
                "field": spec["Field"],
                "display_name": spec["Name"],
                "kind": spec["Kind"],
                "importance": spec["Importance"],
                "tolerance_percent": spec.get("TolerancePercent"),
                "unit": spec.get("Unit", ""),
            }
            for spec in DEFAULT_BUSINESS_SPECS
        ],
    )


@app.get("/source-quality", response_model=SourceQualityResponse)
async def get_source_quality():

    _require_ready()

    source_quality = recommender.source_quality_metrics or {}

    return SourceQualityResponse(
        version=API_VERSION,
        selected_mode=recommender.selected_mode,
        mode_reason=recommender.mode_reason,
        source_quality=source_quality,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_analogs(request: PredictRequest):

    _require_ready()

    import asyncio

    start_time = time.time()

    query_id = recommender.find_query_id(request.query)

    if query_id is None:
        return PredictResponse(
            query=request.query,
            query_id=None,
            query_type="not_found",
            k=request.k,
            results=[],
            total_candidates_generated=0,
            total_candidates_validated=0,
            total_candidates_passed=0,
            catalog_size=recommender.catalog_size,
            execution_time_ms=(time.time() - start_time) * 1000.0,
            timestamp=datetime.now().isoformat(),
        )

    service_result = await asyncio.to_thread(
        recommender.get_analogs_by_id,
        query_id,
        request.k,
        request.min_score,
        request.include_details,
    )

    query_type = _detect_query_type(
        request_query=request.query,
        query_id=query_id,
        service_result=service_result,
    )

    results = [
        _to_analog_item(item, request.include_details)
        for item in service_result["results"]
    ]

    return PredictResponse(
        query=request.query,
        query_id=query_id,
        query_type=query_type,
        k=request.k,
        results=results,
        total_candidates_generated=int(service_result.get("total_candidates_generated", 0)),
        total_candidates_validated=int(service_result.get("total_candidates_validated", 0)),
        total_candidates_passed=int(service_result.get("total_candidates_passed", len(results))),
        catalog_size=recommender.catalog_size,
        execution_time_ms=(time.time() - start_time) * 1000.0,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict_by_id", response_model=PredictResponse)
async def predict_analogs_by_id(request: PredictByIdRequest):

    _require_ready()

    import asyncio

    start_time = time.time()

    if request.product_id not in recommender.id_to_idx:
        raise HTTPException(status_code=404, detail="Товар с таким ID не найден")

    service_result = await asyncio.to_thread(
        recommender.get_analogs_by_id,
        request.product_id,
        request.k,
        request.min_score,
        request.include_details,
    )

    results = [
        _to_analog_item(item, request.include_details)
        for item in service_result["results"]
    ]

    return PredictResponse(
        query=request.product_id,
        query_id=request.product_id,
        query_type="id",
        k=request.k,
        results=results,
        total_candidates_generated=int(service_result.get("total_candidates_generated", 0)),
        total_candidates_validated=int(service_result.get("total_candidates_validated", 0)),
        total_candidates_passed=int(service_result.get("total_candidates_passed", len(results))),
        catalog_size=recommender.catalog_size,
        execution_time_ms=(time.time() - start_time) * 1000.0,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():

    if not system_ready or recommender is None:
        return StatsResponse(
            initialized=False,
            catalog_size=0,
            selected_mode="not_initialized",
            mode_reason="Система не инициализирована",
            business_weights={},
            valid_coverage={},
            source_quality=None,
            analogs_pass_rate=None,
            analogs_diagnostics={},
        )

    stats = recommender.get_stats()

    return StatsResponse(
        initialized=True,
        catalog_size=stats["catalog_size"],
        selected_mode=stats["selected_mode"],
        mode_reason=stats.get("mode_reason"),
        business_weights=stats["business_weights"],
        valid_coverage=stats["valid_coverage"],
        source_quality=stats.get("source_quality"),
        analogs_pass_rate=stats.get("analogs_pass_rate"),
        analogs_diagnostics=stats.get("analogs_diagnostics", {}),
    )


@app.get("/categories")
async def get_categories():

    return {
        "categories": [
            {"id": "split_systems", "name": "Сплит-системы"},
            {"id": "multisplit", "name": "Мульти-сплит системы"},
            {"id": "cassette", "name": "Кассетные кондиционеры"},
            {"id": "floor_ceiling", "name": "Напольно-потолочные"},
            {"id": "ducted", "name": "Канальные кондиционеры"},
        ]
    }


def run_server():
    """Запускает сервер."""
    uvicorn.run(
        "src.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
