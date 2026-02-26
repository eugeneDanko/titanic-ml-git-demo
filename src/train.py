import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.config import (
    DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    BEST_MODEL_DIR

)
from src.preprocessing import create_preprocessing_pipeline


def load_data():
    df = pd.read_csv(DATA_DIR / 'raw' / 'train.csv')
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X,y

def split_data(X, y):
    return train_test_split(X,y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def build_pipeline():
    preprocess = create_preprocessing_pipeline()

    model = LogisticRegression(random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[('preprocessing', preprocess), ('model', model)])

    return pipeline

def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_pipeline(pipeline, X_test, y_test):

    y_pred = pipeline.predict(X_test)

    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred),
               'f1': f1_score(y_test, y_pred)}
    return metrics

def save_artifacts(pipeline, metrics, version = 'v1'):
    model_dir = BEST_MODEL_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Модель сохранена: {model_path}")
    print(f"Метрики сохранены: {metrics_path}")

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)

    pip_proc = build_pipeline()
    pipeline = train_pipeline(pip_proc, X_train, y_train)

    metrics = evaluate_pipeline(pipeline, X_test, y_test)

    for name, i in metrics.items():
        print(f'{name}: {i:.3f}')
    save_artifacts(pipeline, metrics)

if __name__ == '__main__':
    main()
