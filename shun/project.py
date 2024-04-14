# loan approval prediction 
# using logistic regression, support vector machine , decision tree
# then compairing their accuracy and using the best model to predict the loan approval

import pandas as pd
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Join the current directory with the relative path to the CSV file
data_path = os.path.join(current_dir, 'loan_data.csv')

# Load the data
data = pd.read_csv(data_path)

print(data.head())
