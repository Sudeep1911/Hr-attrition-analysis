import pickle
from flask import Flask, render_template, request
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask_cors import CORS
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import models
import modelpic
from sklearn.preprocessing import LabelEncoder
# Load the trained model from a pickle file
with open('dt_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
app_flask = Flask(__name__)
CORS(app_flask)

# Define Flask routes

@app_flask.route('/')
def index():
    features=["Age","BusinessTravel","Department","DistanceFromHome","Education","EducationField","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
    def prediction(fdata):
        pre=model.predict([fdata])
        contributions = fdata * model.coef_[0]
        
        max_feature = features[np.argmax(np.abs(contributions))]
        max_contribution = contributions[np.argmax(np.abs(contributions))]

        return([pre[0],max_feature,max_contribution])
    data=pd.read_csv("HR-Employee-Attrition.csv")
    categorical_column = ['Attrition', 'BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime','EducationField']
    encoder=LabelEncoder()
    data[categorical_column]=data[categorical_column].apply(encoder.fit_transform)

    data = data.drop(['MonthlyRate', 'DailyRate','HourlyRate','Over18','EmployeeCount','StandardHours','EmployeeNumber'], axis=1)
    fdata = data[(data['Attrition'] == 1) & (data['PerformanceRating'] == 4)]
    y_fdata=fdata['Attrition']
    fdata=fdata.drop(['Attrition'],axis=1)
    predicted_values = []
    affected_features=[]
    contri=[]

    # Iterate over rows and apply the prediction function
    for index, row in fdata.iterrows():
        predicted_value = prediction(row)
        predicted_values.append(predicted_value[0])
        affected_features.append(predicted_value[1])
        contri.append(predicted_value[2])
    fdata["Predicted Attrition"],fdata["Feature affecting"],fdata["Contribution"]=predicted_values,affected_features,contri

    dataframe=fdata[fdata['Predicted Attrition']==1].iloc[1:]
    
    
    return render_template('dataframe.html',df=dataframe)

if __name__ == '__main__':
    app_flask.run(debug=True)
