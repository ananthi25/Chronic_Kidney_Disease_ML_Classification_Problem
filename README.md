# Chronic_Kidney_Disease_ML_Classification_Problem

# 🩺 Chronic Kidney Disease Prediction

## 📌 Project Overview

This project builds a **Machine Learning classification model to predict Chronic Kidney Disease (CKD)** based on patient medical attributes.

The project follows a complete end-to-end Machine Learning workflow:

**Data Understanding → Data Preprocessing → Missing Value Treatment → EDA → Encoding → Train/Test Split → Model Training → Prediction → Model Evaluation**

The primary Machine Learning algorithm used in this project is the **Random Forest Classifier**.

---

## 🎯 Business Problem

Chronic Kidney Disease is a serious medical condition that can progress gradually and may not always be detected at an early stage.

The objective of this project is to build a classification model that predicts whether a patient is:

* `CKD`
* `Not CKD`

The target variable is converted into binary format:

```text
CKD     → 1
Not CKD → 0
```

Therefore, this project is a **Binary Classification** problem.

> **Note:** This project is intended for Machine Learning practice and educational purposes. It is not a medical diagnostic system.

---

## 📊 Dataset

The project uses a Chronic Kidney Disease dataset containing patient-related medical information.

The dataset contains both numerical and categorical features.

### Numerical Features

Examples include:

* Age
* Blood Pressure
* Specific Gravity
* Albumin
* Sugar
* Blood Glucose
* Blood Urea
* Serum Creatinine
* Sodium
* Potassium
* Hemoglobin
* Packed Cell Volume
* White Blood Cell Count
* Red Blood Cell Count

### Categorical Features

Examples include:

* Red Blood Cells
* Pus Cell
* Pus Cell Clumps
* Bacteria
* Hypertension
* Diabetes Mellitus
* Coronary Artery Disease
* Appetite
* Pedal Edema
* Anemia

### Target Variable

```text
classification
```

Possible values:

```text
ckd
notckd
```

---

# 🔄 Machine Learning Workflow

```text
Raw CKD Dataset
       ↓
Data Inspection
       ↓
Identify Numerical & Categorical Columns
       ↓
Missing Value Treatment
       ↓
Remove ID Column
       ↓
Separate Features and Target
       ↓
Target Encoding
       ↓
Feature Classification
       ↓
Exploratory Data Analysis
       ↓
Categorical Encoding
       ↓
Train/Test Split
       ↓
Random Forest Classifier
       ↓
Prediction
       ↓
Model Evaluation
```

---

# 1️⃣ Data Type Identification

The first step is to identify categorical and numerical columns.

### Object/Categorical Columns

```python
object_columns = df.select_dtypes(include=['object']).columns

print("Object Data Type Columns:")
print(object_columns)
```

### Numerical Columns

```python
numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

print("Numerical Data Type Columns:")
print(numerical_columns)
```

### Why?

Different types of data require different preprocessing techniques.

For example:

```text
Numerical Data   → Median / Mean
Categorical Data → Most Frequent / Mode
```

---

# 2️⃣ Missing Value Treatment

Real-world healthcare datasets may contain missing values.

Instead of simply deleting records with missing values, this project uses `SimpleImputer` to replace missing values.

### Categorical Data

```python
from sklearn.impute import SimpleImputer

imp_1 = SimpleImputer(
    missing_values=np.nan,
    strategy="most_frequent"
)
```

The `most_frequent` strategy replaces missing categorical values with the category that occurs most frequently.

### Numerical Data

```python
imp_2 = SimpleImputer(
    missing_values=np.nan,
    strategy="median"
)
```

The median strategy is used to replace missing numerical values.

### Why Median?

Median is less affected by extreme values compared with the mean.

This makes it a useful option when numerical data may contain outliers.

---

# 3️⃣ Apply Imputation

Categorical columns are transformed using:

```python
df_imp_1 = pd.DataFrame(
    imp_1.fit_transform(df[object_columns])
)

df_imp_1.columns = df[object_columns].columns
```

Numerical columns are transformed using:

```python
df_imp_2 = pd.DataFrame(
    imp_2.fit_transform(df[numerical_columns])
)

df_imp_2.columns = df[numerical_columns].columns
```

### `fit_transform()`

This performs two operations:

```text
fit       → Learn the required replacement value
transform → Replace the missing values
```

---

# 4️⃣ Verify Missing Values

After imputation, missing values are checked:

```python
print(df_imp_1.isnull().sum())
print()
print(df_imp_2.isnull().sum())
```

The purpose is to verify that the required missing values have been handled successfully.

---

# 5️⃣ Remove ID Column

The patient ID is not considered a useful predictive feature.

Therefore, it is removed:

```python
df_imp_2 = df_imp_2.iloc[:, 1:]
```

### Why remove ID?

An ID is generally used only to uniquely identify a record.

It does not represent a meaningful medical characteristic of the patient.

Including such an identifier may introduce irrelevant patterns into the Machine Learning model.

---

# 6️⃣ Combine Processed Data

The processed categorical and numerical DataFrames are combined column-wise:

```python
df_imp_1['test'] = 'test'
df_imp_2['test'] = 'test'

table_df = pd.concat(
    [df_imp_1, df_imp_2],
    axis=1
)
```

The temporary helper column is then removed:

```python
table_df = table_df.drop(
    ['test'],
    axis=1
)
```

The resulting DataFrame contains the processed dataset.

---

# 7️⃣ Separate Features and Target

The target variable is:

```text
classification
```

The independent variables/features are separated using:

```python
x = table_df.drop(
    'classification',
    axis=1
)
```

The dependent variable/target is:

```python
y = table_df['classification']
```

Therefore:

```text
X → Patient medical information
y → CKD / Not CKD
```

A simple way to remember:

```text
X = What the model sees
y = What the model predicts
```

---

# 8️⃣ Check Target Distribution

Before building the model, the distribution of the target variable is checked:

```python
y.value_counts()
```

This helps determine how many observations belong to each class.

For example:

```text
ckd       250
notckd    150
```

If one class is significantly larger than another, the dataset may have **class imbalance**.

---

# 9️⃣ Convert Target into Binary Values

The target variable is converted from text labels into numerical labels:

```python
y = np.where(
    y == 'ckd',
    1,
    0
)
```

The resulting representation is:

```text
CKD     → 1
Not CKD → 0
```

The distribution can be verified using:

```python
pd.DataFrame(y).value_counts()
```

This converts the problem into a **binary classification task**.

---

# 🔟 Feature Classification

The features are categorized into:

```text
Categorical Features
Non-Categorical Features
Discrete Features
Continuous Features
```

The purpose is to understand the type and structure of each feature.

Example:

```text
Categorical:
Hypertension
Diabetes Mellitus
Anemia

Continuous:
Blood Pressure
Serum Creatinine
Hemoglobin
```

---

# 1️⃣1️⃣ Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the dataset before Machine Learning model development.

The project uses **D-Tale**:

```python
import dtale

dtale.show(df)
```

EDA helps investigate:

* Missing values
* Duplicate records
* Data distributions
* Outliers
* Feature relationships
* Categorical distributions
* Numerical distributions
* Target-class distribution

### Why EDA?

EDA helps identify data quality issues and patterns before training the Machine Learning model.

---

# 1️⃣2️⃣ One-Hot Encoding

Machine Learning algorithms generally require numerical input.

Categorical variables are converted into numerical variables using One-Hot Encoding:

```python
df_dummies = pd.get_dummies(
    x[categorical_features],
    drop_first=True
)
```

For example:

```text
Hypertension

yes
no
yes
no
```

can become:

```text
Hypertension_yes

1
0
1
0
```

---

## Why `drop_first=True`?

Consider:

```text
Gender

Male
Female
```

Without dropping the first category:

```text
Male   Female
1       0
0       1
```

One column can be inferred from the other.

With:

```python
drop_first=True
```

only one encoded column is retained.

This reduces redundant information.

---

# 1️⃣3️⃣ Convert Boolean Values to 0 and 1

The generated dummy variables may contain Boolean values:

```text
True
False
```

They are converted into numerical values:

```python
df_dummies1 = np.where(
    df_dummies[0:] == True,
    1,
    0
)
```

Then converted back into a DataFrame:

```python
df_dummies1 = pd.DataFrame(
    df_dummies1
)

df_dummies1.columns = df_dummies.columns
```

---

# 1️⃣4️⃣ Combine Encoded and Numerical Features

The original categorical columns are removed and replaced with their encoded versions:

```python
x1 = pd.concat(
    [
        x.drop(columns=categorical_features),
        df_dummies1
    ],
    axis=1
)
```

The resulting `x1` contains numerical data suitable for the Machine Learning model.

---

# 1️⃣5️⃣ Train/Test Split

The dataset is divided into training and testing datasets:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x1,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Dataset Distribution

```text
80% → Training Data
20% → Testing Data
```

### Training Data

Used to train the Machine Learning model.

### Testing Data

Used to evaluate how well the model performs on unseen data.

---

## `random_state=42`

`random_state` ensures that the same random split can be reproduced when the code is executed again.

The value `42` is not mandatory. Any fixed integer can be used.

---

## `stratify=y`

`stratify=y` helps preserve the target-class distribution in both training and testing datasets.

This is especially useful when the target classes are imbalanced.

---

# 1️⃣6️⃣ Random Forest Classifier

The primary Machine Learning algorithm used in this project is **Random Forest Classifier**.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
```

Random Forest is an **ensemble learning algorithm** that combines multiple decision trees.

Conceptually:

```text
                 Training Data
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    Decision Tree  Decision Tree  Decision Tree
        ↓              ↓              ↓
        └──────────────┼──────────────┘
                       ↓
                Final Prediction
```

For classification, multiple trees contribute to the final prediction.

---

# 1️⃣7️⃣ Train the Model

The Random Forest model is trained using:

```python
model.fit(
    x_train,
    y_train
)
```

Here:

```text
x_train → Training features
y_train → Actual training labels
```

The model learns patterns between patient features and the CKD classification.

---

# 1️⃣8️⃣ Generate Predictions

Predictions are generated using the test dataset:

```python
y_prediction = model.predict(
    x_test
)
```

The model produces predictions such as:

```text
1 → CKD
0 → Not CKD
```

---

# 1️⃣9️⃣ Model Evaluation

The model accuracy is calculated using:

```python
from sklearn.metrics import accuracy_score

accuracy_score(
    y_test,
    y_prediction
)
```

Accuracy represents the proportion of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

# 2️⃣0️⃣ Classification Report

A detailed classification report is generated using:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_prediction
    )
)
```

The report provides:

```text
Precision
Recall
F1-score
Support
```

---

# 📈 Evaluation Metrics

## Accuracy

Accuracy measures the percentage of total predictions that were correct.

For example:

```text
100 predictions
95 correct

Accuracy = 95%
```

However, accuracy alone may not be sufficient when the dataset is imbalanced.

---

## Precision

Precision answers:

> Out of all observations predicted as CKD, how many were actually CKD?

High precision means fewer false-positive predictions.

---

## Recall

Recall answers:

> Out of all actual CKD cases, how many did the model correctly identify?

Recall is particularly important in healthcare classification because missing an actual positive case can be more serious than incorrectly identifying a negative case as positive.

---

## F1-Score

F1-score provides a balance between precision and recall.

It is useful when both false positives and false negatives need to be considered.

---

## Support

Support represents the number of actual observations belonging to each class in the evaluation dataset.

---

# 🧪 Example Classification Report

A model may produce results similar to:

```text
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       904
           1       1.00      0.86      0.93        96

    accuracy                           0.99      1000
   macro avg       0.99      0.93      0.96      1000
weighted avg       0.99      0.99      0.99      1000
```

The important point is to evaluate the individual class metrics rather than looking only at accuracy.

For example, if class `1` represents CKD, its **recall** tells us how effectively the model identifies actual CKD cases.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Python
* Pandas
* NumPy
* Data Cleaning
* Missing Value Imputation
* Numerical Data Processing
* Categorical Data Processing
* Exploratory Data Analysis
* D-Tale
* One-Hot Encoding
* Feature Classification
* Binary Classification
* Train/Test Split
* Stratified Sampling
* Random Forest Classification
* Model Training
* Model Prediction
* Model Evaluation
* Precision
* Recall
* F1-score
* Accuracy

---

# 📁 Project Structure

A suggested GitHub repository structure:

```text
chronic-kidney-disease-prediction/
│
├── README.md
│
├── data/
│   └── kidney_disease.csv
│
├── notebooks/
│   └── CKD_Classification.ipynb
│
├── src/
│   └── ckd_model.py
│
├── results/
│   └── model_evaluation.txt
│
└── requirements.txt
```

---

# ⚙️ Installation

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib dtale
```

For Google Colab:

```python
!pip install pandas numpy scikit-learn matplotlib dtale
```

---

# 🚀 How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <your-repository-url>
```

### Step 2 — Open the notebook

Open:

```text
notebooks/CKD_Classification.ipynb
```

### Step 3 — Load the dataset

Place the CKD dataset inside:

```text
data/
```

### Step 4 — Perform preprocessing

Run the preprocessing workflow:

```text
Identify data types
        ↓
Handle missing values
        ↓
Remove ID
        ↓
Separate X and y
        ↓
Encode target
        ↓
Encode categorical features
```

### Step 5 — Split the dataset

Use an 80/20 train-test split with stratification.

### Step 6 — Train Random Forest

Train the classifier using the training dataset.

### Step 7 — Generate predictions

Predict the target values for the test dataset.

### Step 8 — Evaluate the model

Review:

```text
Accuracy
Precision
Recall
F1-score
```

---

# 🔮 Future Improvements

The project can be further improved by implementing:

* Hyperparameter tuning
* Cross-validation
* Confusion matrix
* ROC curve
* AUC score
* Feature importance analysis
* Logistic Regression
* Decision Tree Classifier
* Gradient Boosting
* XGBoost
* Class imbalance handling
* Feature scaling where appropriate
* Scikit-learn Pipeline
* Model serialization using Joblib
* Model deployment using FastAPI or Streamlit

---

# 📌 Project Takeaway

This project demonstrates how a raw healthcare dataset can be converted into a Machine Learning-ready dataset and used to build a binary classification model.

The complete workflow is:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Missing Value Treatment
     ↓
EDA
     ↓
Feature Encoding
     ↓
Train/Test Split
     ↓
Random Forest
     ↓
Prediction
     ↓
Evaluation
```

The main objective is not only to achieve a high accuracy score but also to understand **why each preprocessing, modelling, and evaluation step is required**.

---

# 👩‍💻 Author

**Ananthi P**

Machine Learning | GenAI 

---

## ⭐ Project Summary

| Component            | Details                               |
| -------------------- | ------------------------------------- |
| Problem              | Chronic Kidney Disease Prediction     |
| ML Type              | Binary Classification                 |
| Dataset              | Chronic Kidney Disease                |
| Preprocessing        | Missing Value Imputation              |
| Categorical Handling | One-Hot Encoding                      |
| EDA                  | D-Tale                                |
| Train/Test Split     | 80/20                                 |
| Sampling             | Stratified                            |
| Algorithm            | Random Forest Classifier              |
| Evaluation           | Accuracy, Precision, Recall, F1-score |
| Libraries            | Pandas, NumPy, Scikit-learn, D-Tale   |
