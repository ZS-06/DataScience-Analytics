import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('iris')

# Display structure
print("Shape of dataset:", df.shape)
print("Column names:", df.columns)
print("First 5 rows:")
print(df.head())

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 🔹 Scatter Plot
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', ax=axes[0])
axes[0].set_title("Scatter Plot")

# 🔹 Histogram
axes[1].hist(df['petal_length'], bins=20)
axes[1].set_title("Histogram")
axes[1].set_xlabel("Petal Length")
axes[1].set_ylabel("Frequency")

# 🔹 Box Plot
sns.boxplot(data=df, ax=axes[2])
axes[2].set_title("Box Plot")

# Adjust layout
plt.tight_layout()

# Show all plots together
plt.show()