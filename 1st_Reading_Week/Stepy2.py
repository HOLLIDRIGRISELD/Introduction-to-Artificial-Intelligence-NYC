import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load data and handle missing values for selected features
df = pd.read_csv('train.csv')
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
X = df[features].copy()
X.fillna(X.median(), inplace=True)


# Convert continuous SalePrice into 3 equal categories (Cheap, Medium, Expensive)
threshold_33 = np.percentile(df['SalePrice'], 33.33)
threshold_66 = np.percentile(df['SalePrice'], 66.67)

conditions = [(df['SalePrice'] <= threshold_33),(df['SalePrice'] > threshold_33) & (df['SalePrice'] <= threshold_66),(df['SalePrice'] > threshold_66)]

choices = ['Cheap', 'Medium', 'Expensive']
y = np.select(conditions, choices)


# Split data and scale features (required for Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, log_preds) * 100:.2f}%")
print(f"F1 Score (Macro): {f1_score(y_test, log_preds, average='macro'):.4f}\n")

# Train and evaluate Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")
print(f"F1 Score (Macro): {f1_score(y_test, rf_preds, average='macro'):.4f}\n")

print("Feature Importances (Random Forest):")
print(pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False))

# Plot Confusion Matrices for both models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, log_preds, labels=['Cheap', 'Medium', 'Expensive']), 
            annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Cheap', 'Medium', 'Expensive'], 
            yticklabels=['Cheap', 'Medium', 'Expensive'])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

sns.heatmap(confusion_matrix(y_test, rf_preds, labels=['Cheap', 'Medium', 'Expensive']), 
            annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Cheap', 'Medium', 'Expensive'], 
            yticklabels=['Cheap', 'Medium', 'Expensive'])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.show()