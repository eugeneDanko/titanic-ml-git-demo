import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessing_pipeline
from src.config import *
from src.train_cv_evaluate import data_load
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


PARAM_DISTRIBUTIONS = {

    "GradientBoostingClassifier": {
        "model__n_estimators": randint(100, 500),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth": randint(2, 6),
        "model__min_samples_leaf": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4)
    }
}



def load_best_model_name(version="v4"):
    metrics_path = BEST_MODEL_DIR / version / "cv_metrics.csv"
    df = pd.read_csv(metrics_path)

    # Предполагаем, что baseline уже отсортирован по ROC
    best_model_name = df.sort_values(
        "roc_auc_mean", ascending=False
    ).iloc[0]["model"]

    return best_model_name


def run_random_search(model_name, model, X, y, n_iter=50):

    param_dist = PARAM_DISTRIBUTIONS.get(model_name)

    if param_dist is None:
        raise ValueError(f"No param distribution defined for {model_name}")

    pipeline = Pipeline([
        ("preprocessing", create_preprocessing_pipeline()),
        ("model", model)
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2,
        return_train_score=False
    )

    random_search.fit(X, y)

    return random_search

def save_tuning_artifacts(grid, model_name, version="v5"):

    model_dir = BEST_MODEL_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем лучшую модель
    joblib.dump(grid.best_estimator_,
                model_dir / f"{model_name}_tuned.joblib")

    # Сохраняем результаты grid
    grid_results = pd.DataFrame(grid.cv_results_)
    grid_results.to_csv(model_dir / "grid_search_results.csv", index=False)

    # Сохраняем best params
    pd.Series(grid.best_params_).to_csv(
        model_dir / "best_params.csv"
    )

    print("Тюнинг завершён и артефакты сохранены")


def main():

    X, y = data_load()

    # 1. Определяем лучшую baseline модель
    best_model_name = load_best_model_name()

    print(f"Baseline best model: {best_model_name}")

    # 2. Загружаем baseline модель
    model_path = BEST_MODEL_DIR / "v4" / "models" / f"{best_model_name}.joblib"
    model = joblib.load(model_path).named_steps["model"]

    # 3. Запускаем GridSearch
    grid = run_random_search(best_model_name, model, X, y, n_iter=50)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    # 4. Сохраняем
    save_tuning_artifacts(grid, best_model_name)


if __name__ == "__main__":
    main()