"""
Установочный файл.
"""

from pathlib import Path

from setuptools import find_packages, setup


def read_readme():
    """Читает README."""
    readme_path = Path(__file__).parent / "README.md"

    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")

    return ""


setup(
    name="ventilation-recommender",
    version="1.0.0",
    description="Система подбора аналогов вентиляционного и климатического оборудования на базе бизнес-правил.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Data Scientist Student",
    author_email="",
    url="https://github.com/sladeshka/thesis.netology",
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
        "nlp": [
            "pymorphy3>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vent-recommender=src.api:run_server",
        ],
    },
)
