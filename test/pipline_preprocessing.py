import  pandas as pd
from src.preprocessing import create_preprocessing_pipeline
from src.config import *
from  pathlib import Path
path_to_train = DATA_DIR/'raw'/'train.csv'
df = pd.read_csv(path_to_train)

X = df.drop('Survived', axis=1)
y = df['Survived']

pip_proc = create_preprocessing_pipeline()

pip_proc.fit(X)

X_proc = pip_proc.transform(X)

print(f"\n6. Результат:")
print(f"   • Исходный размер: {X.shape}")
print(f"   • После обработки: {X_proc.shape}")
print(f"   • Тип данных: {type(X_proc)}")