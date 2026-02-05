import os
import joblib
import json
import pandas as pd

prj_root = os.path.dirname(os.getcwd())

base_path_to_file = ['data/processed/train_processed.csv',
                    'data/processed/train_features.csv',
                    'data/processed/train_target.csv',
                    'models/scalers/STDscaler.joblib',
                    'models/scalers/MINMAXscaler.joblib',
                    'models/metadata/preprocessing.json']


def load_processed_data(path_to_data=None, formt = 'split'):
    if formt == 'split':
        path_to_data = path_to_data if path_to_data else [os.path.join(prj_root, 'data', 'processed', 'train_features.csv'), os.path.join(prj_root, 'data', 'processed', 'train_target.csv')]

        for path in path_to_data:
            if not os.path.exists(path):
                raise FileNotFoundError(f'нет директории/файла: {path}')

        X_train = pd.read_csv(path_to_data[0], encoding='utf-8')
        y_train = pd.read_csv(path_to_data[1], encoding='utf-8').squeeze()
        print(f"Загружено: X.shape={X_train.shape}, y.shape={y_train.shape}")
        return X_train, y_train


    path_to_data = path_to_data if path_to_data else os.path.join(prj_root, 'data', 'processed', 'train_processed.csv')
    if not os.path.exists(path_to_data):
        raise FileNotFoundError(f'нет директории/файла: {path_to_data}')
    return pd.read_csv(path_to_data, encoding='utf-8')



def load_scaler(scale, path_to_scale=None):
    path_to_scale = path_to_scale if path_to_scale else os.path.join(prj_root, 'models', 'scalers')
    if scale == 'std':
        return joblib.load(os.path.join(path_to_scale, 'STDscaler.joblib'))
    elif scale == 'minmax':
        return joblib.load(os.path.join(path_to_scale, 'MINMAXscaler.joblib'))


def load_metadata(path_to_meta=None):
    path_to_meta = path_to_meta if path_to_meta else os.path.join(prj_root, 'models', 'metadata', 'preprocessing.json')
    with open(path_to_meta, 'r', encoding='utf-8') as file:
        return json.load(file)

def check_of_structure(file_to_check=None):
    path_to_file = file_to_check if file_to_check else  base_path_to_file

    miss_path = []
    for file in path_to_file:
        path = os.path.join(prj_root, file)
        if not os.path.exists(path):
            miss_path.append(path)
    if len(miss_path):
        print(f'отсутствующие файлы: {miss_path}')
        return False
    print('все файлы существуют')
    return True