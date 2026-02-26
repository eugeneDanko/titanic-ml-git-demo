import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.train_cv_evaluate import data_load
from src.config import *




def precision_at_n(model, X, y, n):

    probs = model.predict_proba(X)[:, 1]
    sorted_idx = np.argsort(probs)[::-1]

    top_n_idx = sorted_idx[:n]
    y_top_n = y.iloc[top_n_idx]

    precision = y_top_n.sum() / n
    return precision


def lift_at_n(model, X, y, n):

    base_rate = y.mean()
    precision = precision_at_n(model, X, y, n)

    return precision / base_rate



def main():


    X, y = data_load()

    # Holdout split (честная оценка)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )


    model_path = BEST_MODEL_DIR / "v5"
    model_file = list(model_path.glob("*_tuned.joblib"))[0]

    model = joblib.load(model_file)

    model.fit(X_train, y_train)


    probs_test = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs_test)
    pr = average_precision_score(y_test, probs_test)


    percent = int(input()) / 100
    n = int(len(y_test) * percent)

    precision_n = precision_at_n(model, X_test, y_test, n)
    lift_n = lift_at_n(model, X_test, y_test, n)

    print("===== FINAL EVALUATION =====")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr:.4f}")
    print(f"Precision@{percent*100:.0f}%: {precision_n:.4f}")
    print(f"Lift@{percent*100:.0f}%: {lift_n:.2f}")

    model.fit(X, y)

    prod_dir = BEST_MODEL_DIR / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, prod_dir / "final_model.joblib")

    print("Production модель сохранена.")


if __name__ == "__main__":
    main()