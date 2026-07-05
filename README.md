# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project was completed as **Project 2** of the **DecodeLabs Data Science Internship**.

The objective is to build a supervised machine learning pipeline to detect fraudulent credit card transactions. Since fraud datasets are highly imbalanced, the project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data and improve model performance.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Handle class imbalance using SMOTE.
- Train multiple classification models.
- Compare model performance using appropriate evaluation metrics.
- Prevent data leakage using a machine learning pipeline.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 📂 Dataset

- **Dataset:** Credit Card Fraud Detection
- **Source:** Kaggle
- **Target Column:** `Class`
  - `0` → Legitimate Transaction
  - `1` → Fraudulent Transaction

---

## ⚙️ Project Workflow

1. Import libraries
2. Load the dataset
3. Perform Exploratory Data Analysis (EDA)
4. Check missing values and duplicates
5. Analyze class imbalance
6. Split data into training and testing sets
7. Apply SMOTE to the training data
8. Build Logistic Regression pipeline
9. Build Random Forest pipeline
10. Perform hyperparameter tuning using GridSearchCV
11. Evaluate models using:
    - Precision
    - Recall
    - ROC-AUC
    - Confusion Matrix
    - Classification Report
12. Compare model performance

---

## 📊 Machine Learning Models

- Logistic Regression
- Random Forest Classifier

---

## 📈 Evaluation Metrics

- Precision
- Recall
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---


## 🚀 How to Run

1. Open `Project2.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `creditcard.csv`.
3. Run all cells from top to bottom.
4. Review the evaluation metrics and model comparison.

---

## 📚 Key Learning Outcomes

- Supervised Machine Learning
- Fraud Detection
- Handling Imbalanced Data
- SMOTE
- Logistic Regression
- Random Forest
- Hyperparameter Tuning
- GridSearchCV
- Machine Learning Pipelines
- Model Evaluation

---

## 🎓 Internship

**DecodeLabs – Data Science Internship**

Project 2: **Fraud Detection Pipeline**

---

⭐ Thank you for visiting this repository! Feel free to explore the project and share your feedback.
