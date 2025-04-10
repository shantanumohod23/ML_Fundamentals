import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv("../data/heart.csv")

# Initial data checks
print("First 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())

print("\nDataset Info:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

print(f"\nShape: {data.shape}, Size: {data.size}, Dimensions: {data.ndim}")

# Check for missing values
print("\nMissing Values (by column):")
print(data.isna().sum())

# Check and remove duplicates
if data.duplicated().any():
    print("\nDuplicates found. Removing...")
    data = data.drop_duplicates()
else:
    print("\nNo duplicate records found.")

# Splitting dataset
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy Score: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
labels = np.unique(Y_test)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
