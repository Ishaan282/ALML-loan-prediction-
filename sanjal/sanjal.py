# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('D:\OneDrive - MSFT\Study\project\loan prediction\sanjal\hi.csv')
df.drop('Loan_ID',axis=1,inplace=True)
# Preprocess the dataset
# Handle missing values if any

df.fillna(method='ffill', inplace=True)

# Convert categorical variables to numerical values
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
  label_encoders[column] = LabelEncoder()
  df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logistic_regressor = LogisticRegression()

# Create an imputer object (replace 'mean' with 'median' or 'most_frequent' if preferred)
imputer = SimpleImputer(strategy='mean')

# Impute missing values in both training and testing sets
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the classifier
logistic_regressor.fit(X_train_imputed, y_train)

# Make predictions on the test set
y_pred = logistic_regressor.predict(X_test)

# Evaluate model performance (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")

# Calculate ROC AUC (optional)
y_pred_proba = logistic_regressor.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve (optional)
# ... (similar to Decision Tree code)

# Function to take user input and predict loan status
def predict_loan_status():
  input_data = {}
  for column in X.columns:
    # For categorical fields, use the label encoder to transform input to numerical value
    if column in label_encoders:
      user_input = input(f'Enter {column}: ')
      input_data[column] = label_encoders[column].transform([user_input])[0]
    else:
      input_data[column] = float(input(f'Enter {column}: '))

  # Convert input data to DataFrame
  input_df = pd.DataFrame([input_data])

  # Predict loan status based on probability threshold (adjust threshold as needed)
  approval_proba = logistic_regressor.predict_proba(input_df)[:, 1]
  return 'Loan Approved' if approval_proba >= 0.5 else 'Loan Denied'

# Call the function to take user input and predict loan status
loan_status = predict_loan_status()
print(loan_status)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()