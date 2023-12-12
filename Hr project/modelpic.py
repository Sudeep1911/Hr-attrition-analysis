import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def generate_model_results():
    data = pd.read_csv("HR-Employee-Attrition.csv")

    le = LabelEncoder()
    data['BusinessTravel'] = le.fit_transform(data['BusinessTravel'])
    data['Department'] = le.fit_transform(data['Department'])
    data['EducationField'] = le.fit_transform(data['EducationField'])
    data['JobRole'] = le.fit_transform(data['JobRole'])
    data['Gender'] = le.fit_transform(data['Gender'])
    data['MaritalStatus'] = le.fit_transform(data['MaritalStatus'])
    data['OverTime'] = le.fit_transform(data['OverTime'])

    y = data['Attrition']
    X = data.drop(['EmployeeCount', 'Attrition', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(solver='lbfgs'),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=0, max_depth=5, criterion='entropy')
    }

    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve between ML Models')
    plt.legend()
    plt.show()
