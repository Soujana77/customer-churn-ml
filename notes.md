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
Task-4
- Structured ML pipeline (clean → split → scale → train)
- Applied StandardScaler to normalize features
- Trained Logistic Regression model successfully
Task 5 and 6
- Generated predictions using trained model
- Evaluated model using accuracy, precision, recall, and F1-score
- Analyzed confusion matrix to understand model performance
--------
- Achieved ~78% accuracy using Logistic Regression
- Observed lower recall (~51%) meaning many churn cases missed
- Identified need to improve recall for better business impact
05/04/2026
- Implemented Logistic Regression as baseline model
- Evaluated model using accuracy, precision, recall, F1-score
- Implemented Random Forest for comparison
- Observed Random Forest did not improve recall
- Identified need to handle class imbalance
-----------------
- Applied class_weight='balanced' to handle class imbalance
- Improved model performance (higher recall and F1-score)
- Selected balanced Logistic Regression as final model
---------------------
- Saved trained model using joblib
- Saved scaler for consistent preprocessing during predictions
- Model ready for deployment
--------------------
- Improved frontend UI using HTML and CSS
- Added structured form inputs (tenure, monthly charges, contract, internet service)
- Enhanced user experience with clean layout and styled components
- Improved prediction output message formatting
06/04/2026
- Fixed incorrect input handling in Flask app
- Mapped user inputs to correct feature names
- Ensured model receives proper structured input
- Improved prediction reliability
---------------------
- Added rule-based explanation system
- Provided reasons for churn prediction
- Improved interpretability of model output
---------------------
07/04/2026
- Converted single-page app into multi-page Flask application
- Added Home, Dashboard, and Customer Analysis pages
- Implemented navigation using base template
- Integrated ML model into analysis page
- Fixed feature mapping for accurate predictions
- Improved UI using CSS (clean layout, structured inputs)
- Added subscription plan selection instead of manual input
- Added churn prediction with reasons and suggestions