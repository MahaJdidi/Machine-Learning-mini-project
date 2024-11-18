

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle


warnings.filterwarnings("ignore")


data = pd.read_csv("Employee_Attritionmod_cleaned.csv")



X = data[['satisfaction_level', 'average_monthly_hours']].values  
y = data['Yes_attrition'].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


pickle.dump(log_reg, open('model.pkl', 'wb'))

print("Model training completed. Model saved as 'model.pkl'.")
