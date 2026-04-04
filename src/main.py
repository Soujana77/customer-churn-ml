import pandas as pd

# Load dataset
df = pd.read_csv('data/churn.csv')

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df = df.dropna(subset=['TotalCharges'])

# Encode gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Drop customerID (important)
df = df.drop('customerID', axis=1)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Check shape
print(df.shape)

# Preview
print(df.head())
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)