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




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Display the first few rows of the data
print(data.head())

# Display the summary statistics of the data
print(data.describe())

# Display the information of the data
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Plot the distribution of numerical variables
numerical_cols = data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Plot the count of categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data[col])
    plt.title(f'Count of {col}')
    plt.xticks(rotation=90)
    plt.show()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
