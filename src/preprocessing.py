"""
preprocessing.py - обработка данных Titanic
Простая версия для понимания структуры ML проекта
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from pathlib import Path
# Импортируем настройки из config.py
from src.config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, COLUMNS_TO_DROP,
    AGE_IMPUTE_STRATEGY, EMBARKED_IMPUTE_STRATEGY,
    FARE_BINS, FARE_LABELS, AGE_BINS, AGE_LABELS,
    METADATA_DIR, DATA_PATH, TRAIN_FILE
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Класс для создания новых признаков.
    Пример: из имени извлекаем титул, из SibSp+Parch делаем FamilySize.
    Наследуем от BaseEstimator и TransformerMixin,
    чтобы работать в sklearn Pipeline.
    """

    def __init__(self):
        """Инициализация. Пока параметров нет."""
        pass

    def fit(self, X, y=None):
        """
        Метод fit обязателен для Pipeline.
        Здесь ничего не обучаем, просто возвращаем себя.
        """
        return self  # Просто возвращаем объект

    def transform(self, X):
        """
        Основной метод - создает новые признаки.

        Принимает DataFrame, возвращает DataFrame с новыми колонками.
        """
        # Создаем копию, чтобы не менять исходные данные
        X = X.copy()

        # 1. Размер семьи
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        # 2. Пассажир один?
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)


        # 3. Группы возраста (разбиваем на категории)
        X['AgeGroup'] = pd.cut(
            X['Age'],
            bins=AGE_BINS,
            labels=AGE_LABELS
        )

        # 4. Группы цены билета
        X['FareGroup'] = pd.cut(
            X['Fare'],
            bins=FARE_BINS,
            labels=FARE_LABELS
        )
        self.feature_names_out_ = X.columns.tolist()
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Удаляет указанные колонки из данных.
    Простой трансформер для очистки.
    """

    def __init__(self, columns_to_drop):
        """
        Инициализация с списком колонок для удаления.

        Args:
            columns_to_drop: список имен колонок
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        """Фитируем (ничего не делаем)."""
        return self

    def transform(self, X):
        """
        Удаляем колонки.

        Args:
            X: DataFrame

        Returns:
            DataFrame без указанных колонок
        """
        # Удаляем только те колонки, которые есть в данных
        existing_columns = [col for col in self.columns_to_drop
                            if col in X.columns]

        if existing_columns:
            X = X.drop(columns=existing_columns)

        self.feature_names_out_ = X.columns.tolist()
        return X


def create_preprocessing_pipeline():
    """
    Создает полный пайплайн обработки данных.

    Returns:
        sklearn.Pipeline - готовый к обучению пайплайн
    """
    # ШАГ 1: Создание новых признаков
    feature_engineering = FeatureEngineer()

    # ШАГ 2: Удаление ненужных колонок
    drop_columns = DropColumns(columns_to_drop=COLUMNS_TO_DROP)

    # ШАГ 3: Определяем, какие признаки будут после FeatureEngineering

    # Числовые признаки (старые + новые)
    numeric_features = [
        'Age', 'Fare', 'SibSp', 'Parch',  # исходные
        'FamilySize'  # новые
    ]

    # Категориальные признаки (старые + новые)
    categorical_features = [
        'Sex', 'Embarked', 'Pclass',  # исходные
        'AgeGroup', 'FareGroup', 'IsAlone'  # новые
    ]

    # ШАГ 4: Обработка числовых признаков
    numeric_transformer = Pipeline(steps=[
        # Заполняем пропуски медианой
        ('imputer', SimpleImputer(strategy=AGE_IMPUTE_STRATEGY)),
        # Масштабируем к стандартному распределению
        ('scaler', StandardScaler())
    ])

    # ШАГ 5: Обработка категориальных признаков
    categorical_transformer = Pipeline(steps=[
        # Заполняем пропуски самым частым значением
        ('imputer', SimpleImputer(strategy=EMBARKED_IMPUTE_STRATEGY)),
        # Преобразуем в one-hot кодирование
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # ШАГ 6: Объединяем обработчики для разных типов признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # ШАГ 7: Собираем полный пайплайн
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', feature_engineering),
        ('drop_columns', drop_columns),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline


def save_pipeline(pipeline, name="preprocessor"):
    """
    Сохраняет обученный пайплайн в файл.

    Args:
        pipeline: обученный sklearn Pipeline
        name: имя файла (без .joblib)
    """
    filepath = METADATA_DIR / f"{name}.joblib"
    joblib.dump(pipeline, filepath)
    print(f"✓ Пайплайн сохранен: {filepath}")


def load_pipeline(name="preprocessor"):
    """
    Загружает пайплайн из файла.

    Args:
        name: имя файла (без .joblib)

    Returns:
        Загруженный Pipeline
    """
    filepath = METADATA_DIR / f"{name}.joblib"
    pipeline = joblib.load(filepath)
    print(f"✓ Пайплайн загружен: {filepath}")
    return pipeline

if __name__ == "__main__":
    print("Тестирование preprocessing.py")
    X = pd.read_csv(DATA_PATH / TRAIN_FILE).drop('Survived', axis=1)
    pipeline = create_preprocessing_pipeline()
    print(X)
    pipeline.fit(X)

    print("Пайплайн создан успешно!")
    print("Шаги:", [name for name, _ in pipeline.steps])

    ft = pipeline.get_feature_names_out()
    for i, name in enumerate(ft):
        print(f"{i}: {name}")