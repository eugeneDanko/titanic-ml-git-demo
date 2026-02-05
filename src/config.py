"""
Конфигурация препроцессинга на основе EDA анализа
"""
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Настройки препроцессинга на основе EDA"""

    # Для заполнения пропусков (из твоего EDA: Age 20% пропусков)
    AGE_FILL_STRATEGY = 'median'  # или 'mean', 'cluster'
    EMBARKED_FILL_VALUE = 'mode'  # мода из твоего EDA

    # Для кодирования (из твоего MI анализа)
    CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Pclass']
    NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']


    FARE_BINS = [0, 7.91, 14.45, 31.00, 513.32]  # квантили по EDA
    FARE_LABELS = ['low', 'medium', 'high', 'very_high']

    AGE_BINS = [0, 16, 36, 52, 82]
    AGE_LABELS = ['child', 'younger', 'adult', 'old'] # для квантилей по EDA

    CREATE_FAMILY_SIZE = True
    CREATE_IS_ALONE = True

    # Пути к данным
    RAW_DATA_PATH = 'data/raw/train.csv'
    PROCESSED_DATA_PATH = 'data/processed/train_processed.csv'