import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your data
df = pd.read_csv('D:/OneDrive - MSFT/Study/project/loan prediction/shun/hi.csv')

# Define categorical and numeric features
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Convert categorical features to strings
for feature in categorical_features:
    df[feature] = df[feature].astype(str)

# Define preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define your model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Drop the 'Property_Area' and 'Loan_ID' columns from df
df = df.drop(['Property_Area', 'Loan_ID'], axis=1)

# Split your data into features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
model.fit(X_train, y_train)

# Save your model
joblib.dump(model, 'trained_model.joblib')

# Dummy user input
user_input = pd.DataFrame([[
    "Male", "Yes", "0", "Graduate", "No", 
    5000, 0, 100, 360, 1
]], columns=categorical_features + numeric_features)

# Reorder the columns to match the training data
user_input = user_input[X_train.columns]

# Load your model
loaded_model = joblib.load('trained_model.joblib')

# Predict the loan status
prediction = loaded_model.predict(user_input)

print("Predicted Loan Status: ", prediction)