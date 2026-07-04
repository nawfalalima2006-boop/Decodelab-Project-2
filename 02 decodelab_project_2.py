import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

df = pd.read_csv("/content/sample_data/creditcard.csv")

df.head()

df.shape

df.info()



df.isnull().sum()

df.fillna(df.mean(numeric_only=True), inplace=True)

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.shape

df["Class"].value_counts()

(df["Class"].value_counts(normalize=True) * 100).round(2)

df["Class"].unique()

# Keep only valid classes
df = df[df["Class"].isin([0, 1])]

# Convert to integer
df["Class"] = df["Class"].astype(int)

# Verify
print(df["Class"].unique())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))

sns.countplot(
      data=df,
      x="Class",
      hue="Class",
      palette={0:"#4C72B0", 1:"#D62728"},
      legend=False
      )
plt.yscale("log")
plt.title("Distribution of Legitimate and Fraud Transactions", fontsize=14)
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Count (Log Scale)")

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

X = df.drop("Class", axis=1)
y = df["Class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
 )

from sklearn.linear_model import LogisticRegression

baseline_model = LogisticRegression(max_iter=1000)

baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)

from sklearn.metrics import (
      accuracy_score,
      precision_score,
      recall_score,
      roc_auc_score
)
print("Accuracy :", accuracy_score(y_test, baseline_pred))
print("Precision:", precision_score(y_test, baseline_pred))
print("Recall   :", recall_score(y_test, baseline_pred))
print("ROC-AUC  :", roc_auc_score(y_test, baseline_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, baseline_pred)

print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Baseline Model")

plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, baseline_pred))

print("Before SMOTE:")
print(y_train.value_counts())

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42,k_neighbors=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_smote.value_counts())

fig, axes = plt.subplots(1, 2, figsize=(10,4))

sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title("Before SMOTE")

sns.countplot(x=y_train_smote, ax=axes[1])
axes[1].set_title("After SMOTE")

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_smote = scaler.fit_transform(X_train_smote)

X_test_scaled = scaler.transform(X_test)

from imblearn.pipeline import Pipeline

logistic_pipeline = Pipeline([
      ("scaler", StandardScaler()),
          ("smote", SMOTE(random_state=42)),
              ("classifier", LogisticRegression(max_iter=1000, random_state=42))
              ])

logistic_pipeline.fit(X_train, y_train)

lr_pred = logistic_pipeline.predict(X_test)

print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("ROC-AUC  :", roc_auc_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
    )
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(
    logistic_pipeline,
        X_test,
            y_test
            )

plt.show()

from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42))
            ])

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Legitimate","Fraud"],
    yticklabels=["Legitimate","Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
      "classifier__n_estimators": [100],
          "classifier__max_depth": [None, 10],
              "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
      rf_pipeline,
          param_grid=param_grid,
              cv=3,
                  scoring="recall",
                      n_jobs=-1

)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators": [100],
        "classifier__max_depth": [10, None]
        }
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="recall",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(hasattr(grid_search, "best_params_"))

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

best_pred = best_model.predict(X_test)

print("Precision:", precision_score(y_test, best_pred))
print("Recall:", recall_score(y_test, best_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_pred))

print(classification_report(y_test, best_pred))

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

results = pd.DataFrame({
  "Model":["Logistic Regression","Random Forest"],
  "Precision":[precision_score(y_test, lr_pred), precision_score(y_test, best_pred)],
  "Recall":[recall_score(y_test, lr_pred), recall_score(y_test, best_pred)],
  "ROC-AUC":[roc_auc_score(y_test, lr_pred), roc_auc_score(y_test, best_pred)]
  })

results
