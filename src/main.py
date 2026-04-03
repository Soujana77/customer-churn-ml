import pandas as pd

# Load dataset
df = pd.read_csv('data/churn.csv')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df = df.dropna(subset=['TotalCharges'])

# Encode gender column (Male=0, Female=1)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Display first 5 rows
print(df.head())

# Dataset info
print(df.info())