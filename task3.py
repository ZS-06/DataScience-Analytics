# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("Churn_Modelling.csv")

# =========================
# 2. BASIC CLEANING
# =========================

# Remove unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# =========================
# 3. HANDLE MISSING VALUES
# =========================

# Fill numerical missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill categorical missing values with mode
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# =========================
# 4. ENCODE CATEGORICAL DATA
# =========================

# Label Encoding for Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-Hot Encoding for Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# =========================
# 5. VISUALIZATION (EDA)
# =========================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 🔹 Age vs Churn
sns.boxplot(x=df['Exited'], y=df['Age'], ax=axes[0])
axes[0].set_title("Age vs Churn")

# 🔹 Balance Distribution
axes[1].hist(df['Balance'], bins=20)
axes[1].set_title("Balance Distribution")

# 🔹 Credit Score vs Churn
sns.boxplot(x=df['Exited'], y=df['CreditScore'], ax=axes[2])
axes[2].set_title("Credit Score vs Churn")

plt.tight_layout()
plt.show()

# =========================
# 6. SPLIT FEATURES & TARGET
# =========================

X = df.drop('Exited', axis=1)
y = df['Exited']

# =========================
# 7. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 9. TRAIN RANDOM FOREST MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# 10. MAKE PREDICTIONS
# =========================

y_pred = model.predict(X_test_scaled)

# =========================
# 11. EVALUATE MODEL
# =========================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("===== MODEL RESULTS =====")
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)

# =========================
# 12. FEATURE IMPORTANCE
# =========================

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance_df)

# =========================
# 13. FEATURE IMPORTANCE PLOT
# =========================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")
plt.show()