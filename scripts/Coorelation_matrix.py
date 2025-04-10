# 02_Correlation_Heatmap.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("../data/student_scores.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Compute and display the correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap - Student Scores')
plt.show()
