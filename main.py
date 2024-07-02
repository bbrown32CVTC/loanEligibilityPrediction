# This is a ML Python script for predicting loan eligibility using a Logistic Regression algorithm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error

# Store the datasets as DataFrame variables
df = pd.read_csv('loan_data.csv')

# View the first few rows of the DataFrame datasets
print(df.head())

# Number of rows remaining
print("\nRow Count:", df.shape[0])

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Print fields and columns with missing (Null or NaN) data
print(df.isna())

# Removes rows with missing data
df = df.dropna()

# Number of rows remaining
print("\nRows remaining:", df.shape[0])

# Convert categorical values with numerical values using LabelEncoder
le = LabelEncoder()
df.purpose = le.fit_transform(df['purpose'])

# Split the data into features (X) and target (y)
X = df.drop('credit.policy', axis=1)
y = df['credit.policy']

# Create train_test_split of 70% training and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Select numerical features in the training set
numerical_features_train = X_train.select_dtypes(include=['int64', 'float64'])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = X_train.copy()
X_train_scaled[numerical_features_train.columns] = scaler.fit_transform(numerical_features_train)

# Apply the same transformation to the numerical features in the test set
X_test_scaled = X_test.copy()
X_test_scaled[numerical_features_train.columns] = scaler.transform(X_test.select_dtypes(include=['int64', 'float64']))

# Create the logistic regression ML model and train the model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)

print('Intercept: ', logistic_model.intercept_)

# Classify the data with a prediction using the model
y_pred = logistic_model.predict(X_test_scaled)

# Model Accuracy and Evaluation
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'\nMean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

actual_minus_predicted = sum((y_test - y_pred) ** 2)
actual_minus_actual_mean = sum((y_test - y_test.mean()) ** 2)
r2 = 1 - actual_minus_predicted / actual_minus_actual_mean
print('R²:', r2)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Accuracy Percent:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
