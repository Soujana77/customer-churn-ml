import pandas as pd

# ======================
# 1. Load Dataset
# ======================
df = pd.read_csv('data/churn.csv')

# ======================
# 2. Data Cleaning
# ======================

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df = df.dropna(subset=['TotalCharges'])

# Encode gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Drop unnecessary column
df = df.drop('customerID', axis=1)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# ======================
# 3. Feature & Target Split
# ======================
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# ======================
# 4. Train-Test Split
# ======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 5. Feature Scaling
# ======================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# 6. Model Training
# ======================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training completed successfully")