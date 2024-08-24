import os
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_dir = "datasets"
control_file = os.path.join(data_dir, 'control extracted feature.xlsx')
dementia_file = os.path.join(data_dir, 'dementia extracted feature.xlsx')
control_df = pd.read_excel(control_file)
dementia_df = pd.read_excel(dementia_file)

control_df['label'] = 0
dementia_df['label'] = 1

# Combine the data
data = pd.concat([control_df, dementia_df], ignore_index=True)

# Preprocess the data
# Separate features and labels
X = data.drop(columns=['label', 'File Name'])
y = data['label']

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define cross-validation settings
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define model
log_reg_model = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42)

# Perform cross-validation
accuracy_scores = cross_val_score(log_reg_model, X_scaled, y, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(log_reg_model, X_scaled, y, cv=kf, scoring='precision')
recall_scores = cross_val_score(log_reg_model, X_scaled, y, cv=kf, scoring='recall')
f1_scores = cross_val_score(log_reg_model, X_scaled, y, cv=kf, scoring='f1')

# Calculate performance metrics
metrics = {
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy_scores.mean()],
    "Precision": [precision_scores.mean()],
    "Recall": [recall_scores.mean()],
    "F1 Score": [f1_scores.mean()]
}

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Save the metrics DataFrame to an Excel file
metrics_df.to_excel("classification_metrics_with_cv.xlsx", index=False)

# Train the model on the full dataset
log_reg_model.fit(X_scaled, y)

# Create directory for pickle files if it doesn't exist
pickle_dir = "pickle_files"
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

# Save the trained model as a pickle file with a version based on date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
pickle_filename = os.path.join(pickle_dir, f"logistic_regression_model_{current_time}.pkl")

with open(pickle_filename, 'wb') as f:
    pickle.dump(log_reg_model, f)

print(f"Model saved as: {pickle_filename}")
