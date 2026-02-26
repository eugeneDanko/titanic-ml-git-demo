import pandas as pd
import joblib

from pathlib import Path
from src.config import *


def load_model():
    model_path = BEST_MODEL_DIR / "production" / "final_model.joblib"
    return joblib.load(model_path)


def load_test_data():
    test_path = DATA_DIR / "raw" / "test.csv"
    return pd.read_csv(test_path)


def main():


    model = load_model()


    test_df = load_test_data()


    if "PassengerId" not in test_df.columns:
        raise ValueError("В test.csv отсутствует колонка PassengerId")

    ids = test_df["PassengerId"]


    X_test = test_df.drop(columns=["PassengerId"])


    probs = model.predict_proba(X_test)[:, 1]


    results = pd.DataFrame({
        "PassengerId": ids,
        "probability": probs
    })


    results = results.sort_values(
        by="probability",
        ascending=False
    )

    output_dir = BEST_MODEL_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    print("Предсказания успешно сохранены.")


if __name__ == "__main__":
    main()