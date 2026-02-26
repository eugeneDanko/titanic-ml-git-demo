"""
Конфигурация проекта Titanic ML.
Упрощенная версия для понимания структуры.
"""
from pathlib import Path

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"  # если есть
MODELS_PATH = MODELS_DIR  # переименовать для консистентности

# Настройки данных
TARGET_COLUMN = "Survived"
COLUMNS_TO_DROP = ["PassengerId", "Cabin", "Ticket", "Name"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass"]
NUMERICAL_FEATURES = ["Age", "Fare", "SibSp", "Parch"]

# Настройки препроцессинга
AGE_IMPUTE_STRATEGY = "median"
EMBARKED_IMPUTE_STRATEGY = "most_frequent"
FARE_IMPUTE_STRATEGY = "median"

# Бининг из EDA
FARE_BINS = [0, 7.91, 14.45, 31.0, 513.32]
FARE_LABELS = ["very_low", "low", "medium", "high"]

AGE_BINS = [0, 12, 18, 35, 60, 100]
AGE_LABELS = ["child", "teen", "young_adult", "adult", "senior"]

# Настройки моделей
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Пути для сохранения
SCALERS_DIR = MODELS_DIR / "scalers"
METADATA_DIR = MODELS_DIR / "metadata"
BEST_MODEL_DIR = MODELS_DIR / "best_model"

# Создаем директории (если их нет)
for dir_path in [SCALERS_DIR, METADATA_DIR, BEST_MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)