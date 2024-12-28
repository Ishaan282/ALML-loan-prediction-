# Loan Prediction Model

## Description
This project aims to predict loan approval status using various machine learning models. The dataset contains information about loan applicants, including their education level, employment status, income, loan amount, and other relevant features. The models used in this project are Logistic Regression, Support Vector Machine (SVM), and Decision Tree. The best model is selected based on the highest AUC-ROC score.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Colorama

## Team (github links)
- [Ishaan Singla](https://github.com/Ishaan282)
- [Samiksha Singh](https://github.com/SamikshaSingh25)
- [Sameer Chandra](https://github.com/MajesterSmith)
- [Sanjal](https://github.com/SanjalJain)

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Ishaan282/Loan-prediction.git
    cd loan-prediction
    ```

2. open the file in jupyter notebook:
    ```sh
    file :- project.ipynb 
    ```

3. Install the required packages in your system:
    ```sh
    pip install -r <packages>
    ```

## Usage
1. Ensure the dataset [test.csv](http://_vscodecontentref_/0) is in the project directory.

2. Run the starting columns to train the models and evaluate their performance:

3. The script will output the AUC-ROC scores for each model and display the ROC curves. It will also print the name of the model with the highest accuracy.

4. To predict the loan status for a new applicant, modify the input values in the `predict_loan_status` function in the script and run it.
