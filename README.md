# **Regression model, OLS and VIF on Insuranace Data**

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue.svg)  
A project solving the Kaggle competition [Playground Series - Season 4, Episode 12](https://www.kaggle.com/competitions/playground-series-s4e12/overview), focused on predicting whether a customer will take a product (`Premium Amount`) based on demographic and travel-related features.

---

## **Table of Contents**
- [Dataset](#dataset)
- [Objective](#objective)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Approach](#approach)
- [Results](#results)
- [How to Make Predictions](#how-to-make-predictions)
- [Contributing](#contributing)
- [License](#license)

---

## **Dataset**
The dataset for this competition is available [here](https://www.kaggle.com/competitions/playground-series-s4e12/data).

### **Data Details**
- **Training Data (`train.csv`)**: Includes features and the target variable `ProdTaken`.
- **Test Data (`test.csv`)**: Contains features for prediction.
- **Sample Submission File (`sample_submission.csv`)**: Demonstrates the expected format for submissions.

**Target Variable**:  
- `Premium Amount` (1 = Product Taken, 0 = Product Not Taken)

---

## **Objective**
To build a classification model that predicts whether a customer will take a product based on the provided features.

---

## **Project Workflow**
The project follows these steps:
1. **Data Exploration**:
   - Analyzed the dataset to understand distributions, missing values, and relationships.
2. **Data Preprocessing**:
   - Handled missing values using imputation techniques.
   - Encoded categorical variables and scaled numerical features.
   - Reduced multicollinearity using Variance Inflation Factor (VIF).
3. **Feature Engineering**:
   - Added interaction terms and transformed skewed numerical features.
   - Removed irrelevant and redundant columns.
4. **Model Training**:
   - Experimented with various classification models, including:
     - Logistic Regression
     - Random Forest
     - XGBoost
   - Selected the best model using cross-validation and hyperparameter tuning.
5. **Model Evaluation**:
   - Evaluated model performance using metrics such as Accuracy, F1-Score, and AUC-ROC.
6. **Submission**:
   - Generated predictions for the test dataset and prepared the submission file.

---

## **Installation**
Follow these steps to set up and run the project:

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/playground-s4e12.git
cd playground-s4e12
```

### **2. Install dependencies**
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### **3. Download the dataset**
Download the dataset from Kaggle.
Place the files in the data/ directory.

### **Approach**
Data Preprocessing
* Imputed missing values:
* Numerical features: Filled with the median.
* Categorical features: Filled with the mode.
* Encoded categorical variables using Label Encoding.
* Scaled numerical features using StandardScaler.
* Addressed multicollinearity by removing features with high VIF.

### **Modeling**
Trained and evaluated multiple models:
* Logistic Regression for baseline predictions.
* OLS

### **Evaluation Metrics**
* R2 score
* MSE

### **Results**
Best Model: [OLS]
Accuracy: 
* Dep. Variable:         Premium Amount   R-squared:                       0.002
* Model:                            OLS   Adj. R-squared:                  0.002
* Method:                 Least Squares   F-statistic:                     136.9
* Date:                Sun, 26 Jan 2025   Prob (F-statistic):               0.00
* Time:                        12:57:25   Log-Likelihood:            -9.8168e+06
* No. Observations:             1200000   AIC:                         1.963e+07
* Df Residuals:                 1199981   BIC:                         1.963e+07
* Df Model:                          18                                         
* Covariance Type:            nonrobust                                         


### **How to Make Predictions**
To generate predictions for the test dataset:
1. Train the model using the prepared training data.
2. Use the test dataset to predict and save the results:
```bash
# Assuming `model` is your trained model
predictions = model.predict(test)
submission = pd.DataFrame({'ID': test_ids, 'Premium Amount': predictions})
submission.to_csv('submission.csv', index=False)
```

### **Contributing**
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.
