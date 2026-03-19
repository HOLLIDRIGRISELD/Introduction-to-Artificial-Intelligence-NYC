import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Make pandas output look wider and prevent column wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# PART 1: DUMMY DATA PRACTICE
print("\n" + "="*60)
print(" DUMMY DATASET")
print("="*60)
data = {
    "users": ["Bob", "John", "Mike", "Kelly", "Sofia"],
    "country" : ["Greece", "Albania", "France", "USA", "UK"],
    "rating" : [5, 7, 6, 8, 8]
}
df_dummy = pd.DataFrame(data)
print(df_dummy)

print("\n--- Dummy Data Stats ---")
print("Data Types:\n", df_dummy.dtypes)
print(f"Mean Rating: {df_dummy['rating'].mean():.2f}")
print(f"Median Rating: {df_dummy['rating'].median():.2f}")
print(f"Standard Deviation: {df_dummy['rating'].std():.2f}")


# PART 2: HOUSE PRICE DATASET & EDA
print("\n" + "="*60)
print(" HOUSE PRICE PREDICTION DATASET")
print("="*60)

# Load the csv
df = pd.read_csv("House.csv")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Price Statistics ---")
print(f"Mean Price:   ${df['Price'].mean():,.2f}")
print(f"Median Price: ${df['Price'].median():,.2f}")
print(f"Std Dev:      ${df['Price'].std():,.2f}")

print("\n--- Quick Overview (Describe) ---")
print(df.describe().round(2))

print("\n--- Missing Values ---")
print(df.isnull().sum())


# PART 3: MODEL TRAINING
print("\n" + "="*60)
print(" MODEL TRAINING & SPLIT")
print("="*60)

X = df[["YearBuilt","Floors"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# calling the model  
model = LinearRegression()
 # fitting the model 
model.fit(X_train, y_train)

# calculate the predicted price
y_pred = model.predict(X_test)
print("\nFirst 10 Predictions:")
print(y_pred[:10].round(2))


# PART 4: EVALUATION & PLOTTING
print("\n" + "="*60)
print(" MODEL EVALUATION")
print("="*60)

# Calculate the error between the actual and the predicted
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")


# Plot y_pred vs y_actual 
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')

# Draw a perfect prediction line (y = x) in red
# If the model was perfect, all blue dots would sit exactly on this red line
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title("Actual House Prices vs. Predicted House Prices")
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.legend()
plt.grid(True)
plt.show()
