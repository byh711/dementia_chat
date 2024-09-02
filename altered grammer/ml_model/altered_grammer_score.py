import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import pickle

from utils import find_most_recent_file

def calculate_probability(features):
    """
    Calculate the probability using logistic regression weights and feature vector.

    Args:
    features (list or np.array): The new feature vector.
    weights (list or np.array): The weights from the trained logistic regression model.
    scaler (StandardScaler): The scaler used for normalizing the feature vector.

    Returns:
    float: The probability score between 0 and 1.
    """

    current_directory = os.path.dirname(__file__)
    pickle_folder = "pickle_files"
    pickle_path = os.path.join(current_directory, pickle_folder) 
    most_recent_file = find_most_recent_file(pickle_path)

    pickle_file = os.path.join(pickle_path, most_recent_file)
    with open(pickle_file, 'rb') as file:
        logistic_model = pickle.load(file)

    # Extracting the coefficients
    if hasattr(logistic_model, 'coef_'):
        coefficients = logistic_model.coef_
        print("Logistic Regression Coefficients:", coefficients[0])
    else:
        print("The model does not have coefficients.")
    
    # Creating the scalar object using the trained dataset.
    data_dir = "datasets"
    control_file = os.path.join(data_dir, 'control extracted feature.xlsx')
    dementia_file = os.path.join(data_dir, 'dementia extracted feature.xlsx')
    control_df = pd.read_excel(control_file)
    dementia_df = pd.read_excel(dementia_file)
    data = pd.concat([control_df, dementia_df], ignore_index=True)
    X = data.drop(columns=['label', 'File Name'])

    # Normalize the features using the provided scaler
    X.fillna(X.mean(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert input lists to numpy arrays if necessary
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Flatten the scaled features to a 1D array
    features_scaled = features_scaled.flatten()

    # Compute the raw score using the dot product of scaled features and weights
    raw_score = np.dot(features_scaled, coefficients)

    # Apply the logistic function to convert raw score to a probability
    probability = 1 / (1 + np.exp(-raw_score))

    return probability








# # Calculate the probability
probability = calculate_probability(new_features, coefficients)
print(f"Probability: {probability:.2f}")
