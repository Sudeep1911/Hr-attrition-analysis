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
from sklearn.preprocessing import LabelEncoder
# Load the trained model from a pickle file
with open('attrition_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('dt_model.pkl', 'rb') as model_file1:
    model1 = pickle.load(model_file1)
    
    
# Initialize Flask app
app_flask = Flask(__name__)
CORS(app_flask)

# Define Flask routes

@app_flask.route('/')
def index():
    return render_template('index1.html')

@app_flask.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form.get('age'))
        travelclass = int(request.form.get('travelclass'))
        department = int(request.form.get('department'))
        distance = float(request.form.get('distance'))
        edulvl = int(request.form.get('edulvl'))
        edufield = int(request.form.get('edufield'))
        envsatis = int(request.form.get('envsatis'))
        gender = int(request.form.get('gender'))
        jobinvolvement = int(request.form.get('jobinvolvement'))
        joblvl = int(request.form.get('joblvl'))
        jobrole = int(request.form.get('jobrole'))
        jobsatis = int(request.form.get('jobsatis'))
        maritalsatus = int(request.form.get('maritalsatus'))
        nocompanies = int(request.form.get('nocompanies'))
        overtime = int(request.form.get('overtime'))
        performancerat = int(request.form.get('performancerat'))
        relationshiplvl = int(request.form.get('relationshiplvl'))
        stockoption = int(request.form.get('stockoption'))
        experience = int(request.form.get('experience'))
        trainingtime = int(request.form.get('trainingtime'))
        worklifebal = int(request.form.get('worklifebal'))
        yearsworked = int(request.form.get('yearsworked'))
        yrsincurrole = int(request.form.get('yrsincurrole'))
        yrsperformance = int(request.form.get('yrsperformance'))
        yrscurmanager = int(request.form.get('yrscurmanager'))
        salary = int(request.form.get('salary'))
        percenthike = int(request.form.get('percenthike'))
        # Process the form data and make predictions
        
        input_data = [[travelclass,department,distance,edulvl,edufield,envsatis,gender,jobinvolvement,joblvl,jobrole,jobsatis,maritalsatus,salary,nocompanies,overtime,percenthike,performancerat,relationshiplvl,stockoption,experience,trainingtime,worklifebal,yearsworked,yrsincurrole,yrsperformance,yrscurmanager,age]]
        prediction = model.predict(input_data)
        
        decision_values = model.decision_function(input_data)

# Extract probabilities for leave and stay
        probability_leave = 1 / (1 + np.exp(-decision_values))
        probability_stay = 1 - probability_leave
        
        features=["BusinessTravel","Department","DistanceFromHome","Education","EducationField","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","Age"]
        max_feature = features[np.argmax(np.abs(decision_values))]
        max_contri = np.max(np.abs(decision_values))

        # Render the prediction result on the template
        if (prediction[0]==0.0):
            text = f"The Employee Will Stay. Probability of Staying: {float(probability_stay[0]):.2%} | \t Factor Contributing -  {max_feature}:\n{round(float(max_contri * 100), 2)}%"
        else:
            text = f"The Employee Will Leave. Probability of Leaving: {float(probability_leave[0]):.2%} | \t Factor Contributing - {max_feature}\n{round(float(max_contri * 100), 2)}%"

        # Render the prediction result on the template
        return render_template('index1.html', prediction_text=text)

@app_flask.route('/decisiontree')
def decision_tree():
    # Add the path to your decision tree image
    decision_tree_image_path = 'static/tree.png'  # Change this path accordingly
    return render_template('decisiontree.html', decision_tree_image_path=decision_tree_image_path)

@app_flask.route('/analysis1')
def analysis1():
    return render_template('analysis1.html')
@app_flask.route('/empattrition')
def empattrition():
    return render_template('empattrition.html')
@app_flask.route('/models')
def show_models():
    df=models.generate_model_results()
    picture='static/models.png'
    return render_template('models.html',df=df,picture=picture)
@app_flask.route('/dt')
def dt():
    features=["Age","BusinessTravel","Department","DistanceFromHome","Education","EducationField","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
    def prediction(fdata):
        pre=model1.predict([fdata])
        contributions = fdata * model1.coef_[0]
        
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
