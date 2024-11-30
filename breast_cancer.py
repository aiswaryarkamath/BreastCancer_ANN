# breast_cancer_pipeline.py

import pandas as pd
from data_preparation import load_data
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load and Prepare the Dataset
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Step 2: Feature Selection
def select_features(df, num_features=10):
    X = df.iloc[:, :-1]
    y = df['target']
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, y, selected_features

# Step 3: Train ANN Model and Save Artifacts
def train_and_save_model(X, y, selected_features, scaler_file="scaler.pkl", model_file="ann_model.pkl"):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale only the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the ANN model
    model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save the scaler and model
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Scaler saved to {scaler_file}")
    print(f"Model saved to {model_file}")
    return model, scaler

# Main Script
if __name__ == "__main__":
    # Load data
    df = load_data()
    print("Data loaded successfully.")
    
    # Feature selection
    X, y, selected_features = select_features(df, num_features=10)
    print(f"Selected Features:\n{selected_features}")
    
    # Train model and save artifacts
    model, scaler = train_and_save_model(X, y, selected_features)
    print("Model and scaler are ready for use in the Streamlit app.")
