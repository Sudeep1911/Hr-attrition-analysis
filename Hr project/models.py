import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def generate_model_results():
    data=pd.read_csv("HR-Employee-Attrition.csv")

    le = LabelEncoder()
    data['BusinessTravel'] = le.fit_transform(data['BusinessTravel'])
    data['Department'] = le.fit_transform(data['Department'])
    data['EducationField'] = le.fit_transform(data['EducationField'])
    data['JobRole'] = le.fit_transform(data['JobRole'])
    data['Gender'] = le.fit_transform(data['Gender'])
    data['MaritalStatus'] = le.fit_transform(data['MaritalStatus'])
    data['OverTime'] = le.fit_transform(data['OverTime'])

    y=data['Attrition']
    X=data.drop(['EmployeeCount','Attrition','EmployeeNumber','Over18','StandardHours'],axis=1)


    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    models = {
        'Logistic Regression': LogisticRegression(solver='lbfgs'),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=0, max_depth=5, criterion='entropy')
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        avg_roc_auc = np.mean(cv_scores)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        confusion_matrix_data = confusion_matrix(y_test, y_pred)

        classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

        precision = classification_report_dict['macro avg']['precision']
        recall = classification_report_dict['macro avg']['recall']
        f1 = classification_report_dict['macro avg']['f1-score']
        support = classification_report_dict['macro avg']['support']


        results.append({
            'Model Name': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Support': support,
            'Confusion Matrix': confusion_matrix_data,
            'Avg ROC AUC (CV)': avg_roc_auc
        })
    results_df = pd.DataFrame(results)
    return(results_df)