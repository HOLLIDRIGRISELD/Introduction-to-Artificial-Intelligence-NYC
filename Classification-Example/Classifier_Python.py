import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make pandas output look wider and prevent column wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load data
data = pd.read_excel("Diabetes.xlsx")

print("\n" + "="*60)
print(" FIRST 5 ROWS OF DATA")
print("="*60)
print(data.head())

print("\n" + "="*60)
print(" DATA INFO")
print("="*60)
data.info()

print("\n" + "="*60)
print(" DESCRIPTIVE STATISTICS (Rounded)")
print("="*60)
print(data.describe().round(2))

# Treat zeros as missing values and replace with NaN
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
data[columns] = data[columns].replace(0, np.nan)

# Impute missing values with the median
data.fillna(data.median(), inplace=True)

print("\n" + "="*60)
print(" MISSING VALUES AFTER CLEANING")
print("="*60)
print(data.isnull().sum())

# Define features (X) and target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data to prevent the red Convergence Warning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Print the final model results
print("\n" + "="*60)
print(" MODEL RESULTS")
print("="*60)
print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("="*60 + "\n")