import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Missing values check
print("\nMissing values per column:")
print(data.isna().sum())

# Remove duplicates
if data.duplicated().any():
    print("\nDuplicates found. Removing...")
    data = data.drop_duplicates()

# Feature and target separation
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN Model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (default k=5): {accuracy:.2f}")

# Tuning k values
scores = []
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    acc_k = accuracy_score(y_test, y_pred_k)
    scores.append(acc_k)
    print(f"k={k}: Accuracy={acc_k:.2f}")

# Best k visualization (optional)
plt.figure(figsize=(8, 4))
plt.plot(range(1, 10), scores, marker='o')
plt.xlabel("k - Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k")
plt.grid(True)
plt.show()

# Confusion Matrix (for best k)
best_k = scores.index(max(scores)) + 1
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_best_pred = knn_best.predict(X_test)

cm = confusion_matrix(y_test, y_best_pred)
labels = np.unique(y_test)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (k={best_k})")
plt.tight_layout()
plt.show()
