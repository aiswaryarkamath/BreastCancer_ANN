# Breast Cancer Prediction Using Artificial Neural Network (ANN)

This project implements an Artificial Neural Network (ANN) model to predict breast cancer using the **Breast Cancer Dataset** from `sklearn.datasets`. The primary focus is on applying feature selection, hyperparameter tuning using grid search, and building an accurate prediction model.

## Project Overview

Breast cancer is one of the most common cancers worldwide, and early detection is crucial for effective treatment. This project leverages machine learning techniques to build a predictive model that identifies the likelihood of breast cancer based on diagnostic features.

## Features of the Project

- **Dataset**: The Breast Cancer Dataset from the `sklearn.datasets` library.
- **Feature Selection**: To identify and use the most significant features for prediction.
- **Model Architecture**: Artificial Neural Network (ANN) implemented using Python.
- **Hyperparameter Tuning**: Grid search is used to optimize model parameters.
- **Development Environment**: Python with VS Code as the IDE.

## Dataset Details

- **Source**: `sklearn.datasets`
- **Features**: 30 numeric features describing the cell nucleus.
- **Classes**: Binary classification - Malignant (1) or Benign (0).
- **Shape**: 569 samples and 30 features.

### Key Features in the Dataset

- **Mean Radius**
- **Mean Texture**
- **Mean Perimeter**
- **Mean Area**
- ... (and 26 other diagnostic features)

### Target

- `0`: Benign
- `1`: Malignant

## Tools and Libraries Used

- **Programming Language**: Python
- **Libraries**: 
  - `numpy`
  - `pandas`
  - `sklearn`
  - `tensorflow`/`keras`
  - `matplotlib`
  - `seaborn`

## Model Workflow

1. **Data Preprocessing**:
   - Load and explore the dataset.
   - Perform feature scaling and normalization.
   - Apply feature selection techniques to select the most significant features.

2. **Model Building**:
   - Construct an ANN architecture.
   - Train the model on the training data.

3. **Hyperparameter Tuning**:
   - Use Grid Search to optimize hyperparameters (e.g., learning rate, number of neurons, and activation functions).

4. **Evaluation**:
   - Assess the model's performance using metrics like accuracy, precision, recall, and F1-score.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd breast_cancer_prediction
