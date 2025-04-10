import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("../data/Salary_Data.csv")

# Basic checks
print("\nFirst 5 rows:\n", df.head())
print("\nData Info:\n")
print(df.info())
print("\nNull values:\n", df.isnull().sum())

print(f"\nShape: {df.shape}, Size: {df.size}, Dimensions: {df.ndim}")

# Assigning features and target
X = df[["YearsExperience"]]  # X must be a 2D array
y = df["Salary"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Model parameters
m = lr.coef_[0]
c = lr.intercept_
print(f"\nCoefficient (Slope): {m:.2f}")
print(f"Intercept: {c:.2f}")

# Predicting on test data
y_pred = lr.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plotting
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test["YearsExperience"], y=y_test, color="blue", label="Actual")
sns.lineplot(x=X_test["YearsExperience"], y=y_pred, color="red", label="Predicted")
plt.title("Linear Regression - Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
