import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
data=pd.read_csv("HR-Employee-Attrition.csv")
data = data.drop(['MonthlyRate', 'DailyRate','HourlyRate','Over18','EmployeeCount','StandardHours','EmployeeNumber'], axis=1)

le = LabelEncoder()

data['Attrition'] = le.fit_transform(data['Attrition'])
data['BusinessTravel'] = le.fit_transform(data['BusinessTravel'])
data['Department'] = le.fit_transform(data['Department'])
data['EducationField'] = le.fit_transform(data['EducationField'])
data['Gender'] = le.fit_transform(data['Gender'])
data['JobRole'] = le.fit_transform(data['JobRole'])
data['MaritalStatus'] = le.fit_transform(data['MaritalStatus'])
data['OverTime'] = le.fit_transform(data['OverTime'])


features=data.iloc[:,2:]
features['Age']=data.iloc[:,0:1]
target=data.iloc[::,1]

X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

svm=SVC(probability=True)
svm.fit(x_train_scaled,y_train)
filename = 'attrition_model.pkl'
pickle.dump(svm, open(filename, 'wb'))