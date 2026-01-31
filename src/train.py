"""
Basic training script for Titanic dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    """Load Titanic dataset from local CSV."""
    return pd.read_csv("data/raw/train.csv")

def main():
    print("Loading Titanic data...")
    df = load_data()
    print(f"Data shape: {df.shape}")
    
    # Minimal preprocessing
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
    
    # Convert categorical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    X = df[['Pclass', 'Sex', 'Age', 'Fare']]
    y = df['Survived']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {acc:.3f}")
    return model

if __name__ == "__main__":
    model = main()