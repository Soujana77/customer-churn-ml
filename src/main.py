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