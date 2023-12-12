import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import LogisticRegression

import pickle

import graphviz
from dtreeviz.trees import *
from subprocess import check_call
from IPython.display import Image
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree, _tree
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, log_loss, roc_curve, auc
import matplotlib.font_manager

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

features=["Age","BusinessTravel","Department","DistanceFromHome","Education","EducationField","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
def prediction(dt,fdata):
    pre=dt.predict([fdata])
    contributions = fdata * dt.coef_[0]
    
    max_feature = features[np.argmax(np.abs(contributions))]
    max_contribution = contributions[np.argmax(np.abs(contributions))]

    return([pre[0],max_feature,max_contribution])

data1=pd.read_csv("HR-Employee-Attrition.csv")

no_attrition_rows = data1[data1['Attrition'] == 0]

# Delete 900 rows randomly from the selected rows
no_attrition_rows_to_delete = no_attrition_rows.sample(min(900, len(no_attrition_rows)))
data = data1.drop(no_attrition_rows_to_delete.index)

categorical_column = ['Attrition', 'BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime','EducationField']
encoder=LabelEncoder()
data[categorical_column]=data[categorical_column].apply(encoder.fit_transform)

data = data.drop(['MonthlyRate', 'DailyRate','HourlyRate','Over18','EmployeeCount','StandardHours','EmployeeNumber'], axis=1)

fdata = data[(data['Attrition'] == 1) & (data['PerformanceRating'] == 4)]
y_fdata=fdata['Attrition']
fdata=fdata.drop(['Attrition'],axis=1)

Y=data['Attrition']
X=data.drop(['Attrition'],axis=1)

dt = LogisticRegression(max_iter=1000,solver="lbfgs")
dt.fit(X, Y)

predicted_values = []
affected_features=[]
contri=[]

# Iterate over rows and apply the prediction function
for index, row in fdata.iterrows():
    predicted_value = prediction(dt, row)
    predicted_values.append(predicted_value[0])
    affected_features.append(predicted_value[1])
    contri.append(predicted_value[2])
fdata["Predicted Attrition"],fdata["Feature affecting"],fdata["Contribution"]=predicted_values,affected_features,contri

print(fdata[fdata['Predicted Attrition']==1])

filename = 'dt_model.pkl'
pickle.dump(dt, open(filename, 'wb'))
