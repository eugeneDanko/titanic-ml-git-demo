"""
setup.py для проекта Titanic ML
Установка: pip install -e .
"""

from setuptools import setup, find_packages

# Читаем README для длинного описания
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Основная информация
    name="titanic_ml",
    version="0.1.0",
    packages=find_packages(),
    author="Eugene Danko",
    author_email="",

    # Описание
    description="Titanic survival prediction machine learning project",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Ссылки
    url="https://github.com/eugeneDanko/titanic-ml-git-demo",

    # Классификаторы (опционально)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],


    # Зависимости
    install_requires=[
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
    ],

    # Дополнительные зависимости для разработки
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "black>=22.0.0",  # форматирование
            "flake8>=4.0.0",  # линтинг
            "pytest>=7.0.0",  # тесты
        ],
    },

    # Минимальная версия Python
    python_requires=">=3.8",

)