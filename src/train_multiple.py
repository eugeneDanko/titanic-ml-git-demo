import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from  sklearn.linear_model import LogisticRegression
from  sklearn.neighbors import KNeighborsClassifier
from  sklearn.tree import DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    return X,y

def split_data(X, y):
    return train_test_split(X, y, stratify=y, random_state=RANDOM_STATE, test_size=TEST_SIZE)

def models_load():
    return {'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=15, min_samples_leaf=4, random_state=RANDOM_STATE),
            'RandomForestClassifier': RandomForestClassifier(max_depth=5, max_samples=100, random_state=RANDOM_STATE),
            'GradientBoostingClassifier': GradientBoostingClassifier(max_depth=3, random_state=RANDOM_STATE),
            'SVC': SVC(max_iter=1000, probability=True)
            }

def evaluate_model(pipeline, X, y, cv=5):
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring='f1'
    )

    return {'cv_mean_f1': scores.mean(),'cv_std_f1': scores.std()}

def preprocessing_pipeline(model_name, model):
    ppc_pipeline = Pipeline(steps=[('pipeline_preprocessing', create_preprocessing_pipeline()), ('model', model)])
    return ppc_pipeline

def train_pipeline(ppc_pipeline, X_train, y_train):
     ppc_pipeline.fit(X_train, y_train)
     return  ppc_pipeline

def save_artifacts(t_pipelines, m_metrics, version = 'v3'):
    model_dir = BEST_MODEL_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = model_dir / "metrics.csv"

    for m_name in t_pipelines.keys():
        model_file = m_name + '.joblib'
        model_path = model_dir/ 'models'
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(t_pipelines[m_name], model_path/model_file)


    m_metrics.to_csv(metrics_path, index=False)
    print('модели и их метрики сохранены успешно')



def main():
    X,y = data_load()
    cv_results = []
    models = models_load()

    for m_name, model in models.items():
        ppc_pipeline = preprocessing_pipeline('model', model)

        cv_metrics = {'model': m_name}
        cv_metrics = cv_metrics | evaluate_model(ppc_pipeline, X, y)

        cv_results.append(cv_metrics)
    cv_df = pd.DataFrame(cv_results)



    X_train, X_test, y_train, y_test = split_data(X, y)
    ppc_pipelines = {m_name: preprocessing_pipeline(m_name, m) for m_name, m in models.items()}
    t_pipelines = {m_name: train_pipeline(ppc_pipelines[m_name], X_train, y_train) for m_name in models.keys()}



    save_artifacts(t_pipelines, cv_df)

if __name__ == '__main__':
    main()