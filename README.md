# Дипломная работа «Система подбора аналогов вентиляционного оборудования на базе параметризатора и рекомендательных систем для интернет-магазина»

Данный проект был разработан в рамках дипломной работы по специальности «Data Scientist».

Гибридная рекомендательная система для подбора аналогов климатического оборудования.

## Быстрый старт

Создайте виртуальное окружение и активируйте его:

    python -m venv .venv
    source .venv/bin/activate

Для Windows:

    .venv\Scripts\activate

Установите зависимости из `requirements.txt`:

    pip install -r requirements.txt

Установите пакет в editable режиме с dev и nlp extras:

    pip install -e .[dev,nlp]

При необходимости создайте файл `.env` в корне проекта и укажите настройки API:

    HOST=0.0.0.0
    PORT=8000
    DEBUG=false
    DATA_PATH=./data/raw/catalog_products.json

Если `DATA_PATH` не указан или файл не найден, API использует встроенный демо-каталог.

Запустите API из каталога `ventilation_recommender`:

    uvicorn src.api:app --host 0.0.0.0 --port 8000

После запуска доступны эндпоинты:

- `GET /health` — проверка готовности системы.
- `POST /predict` — поиск аналогов по названию товара.
- `POST /predict_by_id` — поиск аналогов по ID товара.
- `GET /stats` — статистика каталога и весов модели.
- `GET /categories` — список категорий.
- `GET /docs` — интерактивная документация FastAPI.

Пример запроса к `/predict`:

    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"query\":\"Daichi Bravo KSGB35\",\"top_k\":5,\"min_score\":0.3}"

Пример запроса к `/predict_by_id`:

    curl -X POST http://localhost:8000/predict_by_id -H "Content-Type: application/json" -d "{\"product_id\":\"prod_0001\",\"top_k\":5,\"min_score\":0.3}"

Запустите тесты:

    pytest
