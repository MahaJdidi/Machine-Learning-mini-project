#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle

# Ignore warnings
warnings.filterwarnings("ignore")

# Load your dataset
data = pd.read_csv("Employee_Attritionmod_cleaned.csv")


# Features and target selection
X = data[['satisfaction_level', 'average_monthly_hours']].values  # Using selected features
y = data['Yes_attrition'].values  # Use the 'Yes_attrition' column as the target (binary: 0 or 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Save the trained model
pickle.dump(log_reg, open('model.pkl', 'wb'))

print("Model training completed. Model saved as 'model.pkl'.")
