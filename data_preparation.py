# data_preparation.py
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
