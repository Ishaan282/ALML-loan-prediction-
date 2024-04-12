import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 

df = pd.read_csv('hi.csv')

df.drop('Loan_ID', axis=1, inplace=True)

# df.fillna(method='ffill', inplace=True)
df.fillna(df.median(),inplace=True)



label_encoders = {}

for column in df.select_dtypes(include=['object']).columns:
  label_encoders[column] = LabelEncoder()
  df[column] = label_encoders[column].fit_transform(df[column])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train)

y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

def predict_loan_status():
  input_data = {}
  for column in X.columns:
    if column in label_encoders:
      while True: 
        try:
          user_input = input(f"Enter {column} (category): ")
          input_data[column] = label_encoders[column].transform([user_input])[0]
          break  
        except ValueError:
          print("Invalid input for", column, ". Please enter a valid category.")
    else:
      while True:  
        try:
          data_type = "numerical" if X.dtypes[column] == np.float64 else "integer"
          user_input = input(f"Enter {column} ({data_type}): ")
          input_data[column] = float(user_input) if data_type == "numerical" else int(user_input)
          break  
        except ValueError:
          print("Invalid input for", column, ". Please enter a", data_type, "value.")


  input_df = pd.DataFrame([input_data])


  approval_proba = dt_classifier.predict_proba(input_df)[:, 1]
  return 'Loan Approved' if approval_proba >= 0.5 else 'Loan Denied'

loan_status = predict_loan_status()
print(loan_status)
