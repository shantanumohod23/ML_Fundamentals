import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv("../data/heart.csv")

# Preview data
print("First 5 rows:\n", data.head())
print("\nLast 5 rows:\n", data.tail())

# Dataset info
print("\nDataset Info:")
print(data.info())

print("\nDescriptive Stats:")
print(data.describe())

print(f"\nShape: {data.shape}, Size: {data.size}, Dimensions: {data.ndim}")

# Check for missing values
print("\nMissing values per column:")
print(data.isna().sum())

# Check for duplicates
if data.duplicated().any():
    print("\nDuplicates found. Removing...")
    data = data.drop_duplicates()
else:
    print("\nNo duplicate rows found.")

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
log = LogisticRegression(max_iter=200)
log.fit(X_train, y_train)

# Predictions
y_pred = log.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
