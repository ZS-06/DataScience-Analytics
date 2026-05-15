import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("loan_approval_dataset.csv")

# =========================
# 2. Clean Column Names
# =========================
df.columns = df.columns.str.strip()

# =========================
# 3. Handle Missing Values
# =========================
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# =========================
# 4. Drop Irrelevant Column
# =========================
if 'loan_id' in df.columns:
    df.drop('loan_id', axis=1, inplace=True)

# =========================
# 5. Convert Categorical → Numerical
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# STEP 2: VISUALIZATION
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x=df['loan_status_ Rejected'], y=df['income_annum'], ax=axes[0])
axes[0].set_title("Income vs Loan Status")

axes[1].hist(df['loan_amount'], bins=20)
axes[1].set_title("Loan Amount Distribution")

sns.boxplot(x=df['loan_status_ Rejected'], y=df['cibil_score'], ax=axes[2])
axes[2].set_title("CIBIL Score vs Loan Status")

plt.tight_layout()
plt.show()

# =========================
# STEP 3: MODEL TRAINING
# =========================

X = df.drop('loan_status_ Rejected', axis=1)
y = df['loan_status_ Rejected']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# =========================
# FINAL RESULTS ONLY
# =========================

print("===== FINAL MODEL COMPARISON =====")

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))