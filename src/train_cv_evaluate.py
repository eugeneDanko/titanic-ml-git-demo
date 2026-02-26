import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from src.preprocessing import create_preprocessing_pipeline
from src.config import *

import joblib
from pathlib import Path


def data_load():
    data_path = DATA_DIR / 'raw' / TRAIN_FILE
    df = pd.read_csv(data_path)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y


def models_load():
    return {'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=15, min_samples_leaf=4,
                                                             random_state=RANDOM_STATE),
            'RandomForestClassifier': RandomForestClassifier(max_depth=5, max_samples=0.8, random_state=RANDOM_STATE),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'SVC': SVC(max_iter=1000, probability=True)
            }


def evaluateCV(pips, X, y, k_fold=5):
    metrics = {'roc_auc': 'roc_auc', 'pr_auc': 'average_precision'}
    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=RANDOM_STATE)

    res = {}

    for name_m, model in pips.items():
        res[name_m] = cross_validate(model, X=X, y=y, cv=cv, scoring=metrics, return_train_score=False)
    return res


def aggregate_results(res_dict):
    rows = []

    for model_name, scores in res_dict.items():
        row = {
            "model": model_name,
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
            "fit_time_mean": scores["fit_time"].mean(),
            "score_time_mean": scores["score_time"].mean()
        }
        rows.append(row)

    return pd.DataFrame(rows).round(5)

def save_artifacts(m_metrics, t_models, version='v4'):
    model_dir = BEST_MODEL_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = model_dir / "cv_metrics.csv"

    for m_name in t_models.keys():
        model_file = m_name + '.joblib'
        model_path = model_dir / 'models'
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(t_models[m_name], model_path / model_file)

    m_metrics.to_csv(metrics_path, index=False)
    print('модели и их метрики сохранены успешно')


def train_model(pips, X, y):
    return {name_m: pips[name_m].fit(X, y) for name_m in pips.keys()}

def select_best_model(
    results_df,
    primary_metric="roc_auc",
    secondary_metric="pr_auc",
    std_threshold=0.05,
    tolerance=0.01
):

    # 1. Фильтр по стабильности
    stable_models = results_df[
        results_df[f"{primary_metric}_std"] <= std_threshold
    ].copy()

    if stable_models.empty:
        raise ValueError("Нет стабильных моделей по заданному порогу std")

    # 2. Сортировка по основной метрике
    stable_models = stable_models.sort_values(
        f"{primary_metric}_mean",
        ascending=False
    )

    best_score = stable_models.iloc[0][f"{primary_metric}_mean"]

    # 3. Отбор моделей, близких к лучшей
    top_models = stable_models[
        stable_models[f"{primary_metric}_mean"] >= best_score - tolerance
    ]

    # 4. Если одна модель — вернуть её
    if len(top_models) == 1:
        return top_models.iloc[0]["model"]

    # 5. Иначе — выбрать по вторичной метрике
    top_models = top_models.sort_values(
        f"{secondary_metric}_mean",
        ascending=False
    )

    return top_models.iloc[0]["model"]
def main():
    X, y = data_load()

    models = models_load()

    pipelines = {
        name: Pipeline([
            ('preprocessing', create_preprocessing_pipeline()),
            ('model', model)
        ])
        for name, model in models.items()
    }

    cv_results_raw = evaluateCV(pipelines, X, y)
    cv_results_df = aggregate_results(cv_results_raw)

    best_model_name = select_best_model(cv_results_df)

    print(f"Best model selected: {best_model_name}")

    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X, y)

    save_artifacts(
        cv_results_df,
        {best_model_name: best_pipeline},
    )


if __name__ == '__main__':
    main()