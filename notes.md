Day 1:

- Loaded dataset using pandas
- Understood DataFrame structure (rows, columns)
- Learned difference between object and numeric types
- Found issue in TotalCharges column (stored as object instead of numeric)
- Converted TotalCharges to numeric using pd.to_numeric()
- Found 11 missing values after conversion
- Removed rows with missing values using dropna()
- Converted TotalCharges from object to numeric
- Identified and removed 11 missing values
- Applied binary encoding to gender column
03/04/2026
- Removed customerID column as it is not useful for prediction
- Applied one-hot encoding to categorical variables
- Converted all features into numeric format for ML models
- Final dataset shape: (7032, 31)
04/04/2026
- Separated features (X) and target (y)
- Performed train-test split (80% training, 20% testing)
- Used random_state=42 for consistent results
- Used correlation to analyze feature importance
- Found top features affecting churn (tenure, contract, charges, internet service)
- Decided to keep all features for model training