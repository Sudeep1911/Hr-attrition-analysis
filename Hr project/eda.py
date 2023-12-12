import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from flask_cors import CORS


app = dash.Dash(__name__, external_stylesheets=["static/style.css"])

server = app.server  # Explicitly create the Flask server instance
CORS(server)  # Enable CORS for all routes

app.external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']


warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

data = pd.read_csv("HR-Employee-Attrition.csv")

data = data.drop(['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'], axis=1)

data2 = data.select_dtypes('int64')
cor_mat = data2.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False

# Create individual figures for each plot
fig1 = px.box(data, x="Attrition", y="Age", points="all")
fig1.update_traces(marker=dict(color='#00CC96'), selector=dict(type='box'))
fig1.update_layout(height=600, width=800)

fig2 = px.box(data, x='JobRole', y='MonthlyIncome', color='Attrition')
fig2.update_traces(marker=dict(color='#FF97FF'), selector=dict(type='box', name='No'))
fig2.update_traces(marker=dict(color='#AB63FA'), selector=dict(type='box', name='Yes'))
fig2.update_xaxes(tickangle=90)
fig2.update_layout(height=500, width=800)

fig3 = px.histogram(data, x="MonthlyIncome", color="Attrition")
fig3.update_layout(height=500, width=800, title="Monthly Income by Attrition", xaxis_title="Monthly Income",
                   yaxis_title="Count")

fig4 = px.scatter_matrix(data,
                         dimensions=["JobLevel", "TotalWorkingYears", "MonthlyIncome", "YearsInCurrentRole",
                                     "PerformanceRating"], color="Attrition")
fig4.update_layout(width=900, height=610, title='Scatter Matrix Plot',)
fig4.update_traces(marker=dict(color='purple'), selector=dict(mode='markers'))

grouped_data = data.groupby(['JobRole', 'EducationField', 'Attrition']).size().reset_index(name='Count')
fig5 = px.bar(grouped_data, x='JobRole', y='Count', color='Attrition', barmode='group', facet_col='EducationField')
fig5.update_layout(title="Attrition by Job Role and Education Field", xaxis_title="Job Role", yaxis_title="Count",)

fig6 = px.box(data, x='BusinessTravel', y='MaritalStatus', color='Attrition')
fig6.update_traces(marker=dict(color='#FF97FF'), selector=dict(type='box', name='No'))
fig6.update_traces(marker=dict(color='#AB63FA'), selector=dict(type='box', name='Yes'))
fig6.update_xaxes(tickangle=90)
fig6.update_layout(height=500, width=1100)

fig7 = px.scatter_matrix(data, dimensions=['MonthlyIncome', 'HourlyRate', 'DistanceFromHome'], color='Attrition',
                         title='Relationship between Monthly Income, Hourly Rate, Distance from Home, and Attrition')
fig7.update_traces(diagonal_visible=False)
fig7.update_layout(height=500, width=800)


heatmap_fig = go.Figure(data=go.Heatmap(
    z=cor_mat.values,
    x=cor_mat.columns,
    y=cor_mat.columns,
    colorscale='Viridis',
    zmin=-1, zmax=1,
    hoverongaps=False,
))
heatmap_fig.update_layout(height=600, width=800)

# Create the bar chart
value_counts = data['Attrition'].value_counts().reset_index()
value_counts.columns = ['Attrition', 'Count']
bar_chart_fig = px.bar(value_counts, x='Attrition', y='Count',
                       labels={'Attrition': 'Attrition', 'Count': 'Count'},
                       title='Attrition plot',
                       color='Attrition',
                       color_discrete_map={'Yes': 'red', 'No': '#FECB52'})
bar_chart_fig.update_traces(marker=dict(line=dict(width=.5, color='DarkSlateGray')))
bar_chart_fig.update_layout(height=400, width=400)

attrition_counts = data['Attrition'].value_counts(normalize=True).reset_index()
attrition_counts.columns = ['Attrition', 'Count']
attrition_counts['Percentage'] = attrition_counts['Count'] * 100
bar_table = go.Figure(data=[go.Table(
    header=dict(values=['Attrition', 'Count', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[attrition_counts['Attrition'], attrition_counts['Count'].round(2), attrition_counts['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])
bar_table.update_layout(title='Summary Table')
bar_table.update_layout(
    title='Summary Table',
    height=300,
    width=400
)

# Create the pie chart for Gender
gender_pie_fig = px.pie(data['Gender'].value_counts().reset_index(),
                         names='Gender',
                         values='count',
                         title='Gender plot',
                         color='Gender',
                         color_discrete_map={'Male': '#B6E880', 'Female': '#FECB52'})
gender_pie_fig.update_layout(height=400, width=400)

gender_counts = data['Gender'].value_counts(normalize=True).reset_index()
gender_counts.columns = ['Gender', 'Count']
gender_counts['Percentage'] = gender_counts['Count'] * 100
gender_table = go.Figure(data=[go.Table(
    header=dict(values=['Gender', 'Count', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[gender_counts['Gender'], gender_counts['Count'], gender_counts['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])
gender_table.update_layout(
    title='Gender Summary Table',
    height=400,
    width=400
)

# Create the pie chart for Marital Status
marital_status_pie_fig = px.pie(data['MaritalStatus'].value_counts().reset_index(),
                                names='MaritalStatus',
                                values='count',
                                title='Marital Status',
                                color='MaritalStatus',
                                color_discrete_map={'Single': '#00CC96', 'Married': '#B6E880', 'Divorced': '#FECB52'})
marital_status_pie_fig.update_layout(height=400, width=400)

marital_status_counts = data['MaritalStatus'].value_counts(normalize=True).reset_index()
marital_status_counts.columns = ['MaritalStatus', 'Count']
marital_status_counts['Percentage'] = marital_status_counts['Count'] * 100
marital_table = go.Figure(data=[go.Table(
    header=dict(values=['Marital Status', 'Count', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[marital_status_counts['MaritalStatus'], marital_status_counts['Count'].round(2), marital_status_counts['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])
marital_table.update_layout(
    title='Marital Status Summary Table',
    height=400,
    width=400
)

# Create the bar chart for Business Travel
frequency_table_business_travel = data['BusinessTravel'].value_counts().reset_index()
frequency_table_business_travel.columns = ['BusinessTravel', 'Frequency']

# Bar chart for Business Travel
business_travel_bar_fig = px.bar(frequency_table_business_travel, x='BusinessTravel', y='Frequency',
                                 title='Business Travel',
                                 color='BusinessTravel',
                                 color_discrete_sequence=['#00CC96', '#B6E880', '#FECB52']
                                 )

business_travel_bar_fig.update_layout(
    height=400,
    width=600,
    yaxis_title='Business Travel',
    xaxis_title='Frequency'
)

frequency_table = data['BusinessTravel'].value_counts(normalize=True).reset_index()
frequency_table.columns = ['BusinessTravel', 'Frequency']
frequency_table['Percentage'] = frequency_table['Frequency'] * 100
busi_table = go.Figure(data=[go.Table(
    header=dict(values=['Business Travel', 'Frequency', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[frequency_table['BusinessTravel'], frequency_table['Frequency'].round(2), frequency_table['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])
busi_table.update_layout(
    title='Business Travel Frequency and Percentage',
    height=400,
    width=400
)

# Create the bar chart for Education Field
frequency_table_education_field = data['EducationField'].value_counts().reset_index()
frequency_table_education_field.columns = ['EducationField', 'Frequency']

# Bar chart for Education Field
education_field_bar_fig = px.bar(frequency_table_education_field, x='EducationField', y='Frequency',
                                title='EducationField',
                                color='EducationField',
                                color_discrete_sequence=['#00CC96', '#B6E880', '#FECB52']
                                )
education_field_bar_fig.update_layout(
    height=400,
    width=600,
    yaxis_title='EducationField',
    xaxis_title='Frequency'
)

frequency_table = data['EducationField'].value_counts(normalize=True).reset_index()
frequency_table.columns = ['EducationField', 'Frequency']
frequency_table['Percentage'] = frequency_table['Frequency'] * 100
edu_table = go.Figure(data=[go.Table(
    header=dict(values=['EducationField', 'Frequency', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[frequency_table['EducationField'], frequency_table['Frequency'].round(2), frequency_table['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])

edu_table.update_layout(
    title='EducationField Frequency and Percentage',
    height=450,
    width=450
)


# Create the pie chart for Department
department_pie_fig = px.pie(data['Department'].value_counts().reset_index(),
                            names='Department',
                            values='count',
                            title='Department',
                            color='Department',
                            color_discrete_map={'Research & Development': '#00CC96', 'Human Resources': '#B6E880', 'Sales': '#FECB52'})

department_pie_fig.update_layout(height=400, width=500)

frequency_table = data['Department'].value_counts(normalize=True).reset_index()
frequency_table.columns = ['Department', 'Frequency']
frequency_table['Percentage'] = frequency_table['Frequency'] * 100
dept_table = go.Figure(data=[go.Table(
    header=dict(values=['Department', 'Frequency', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[frequency_table['Department'], frequency_table['Frequency'].round(2), frequency_table['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])

dept_table.update_layout(
    title='Department Frequency and Percentage Summary',
    height=400,
    width=450
)

# Create the pie chart for Overtime
overtime_pie_fig = px.pie(data['OverTime'].value_counts().reset_index(),
                          names='OverTime',
                          values='count',
                          title='OverTime',
                          color='OverTime',
                          color_discrete_map={'No': '#FECB52', 'Yes': 'red'})

overtime_pie_fig.update_layout(height=400, width=400)

frequency_table = data['OverTime'].value_counts(normalize=True).reset_index()
frequency_table.columns = ['OverTime', 'Frequency']
frequency_table['Percentage'] = frequency_table['Frequency'] * 100
over_table = go.Figure(data=[go.Table(
    header=dict(values=['OverTime', 'Frequency', 'Percentage'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[frequency_table['OverTime'], frequency_table['Frequency'].round(2), frequency_table['Percentage'].round(2).astype(str) + '%'],
               fill_color='lavender',
               align='left'))
])
over_table.update_layout(
    title='OverTime Frequency and Percentage',
    height=400,
    width=400
)


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='HR Analytics Dashboard', children=[html.H1(children='HR Analytics Dashboard'),
            html.Div([
                dcc.Graph(id='bar-chart', figure=bar_chart_fig),
                dcc.Graph(id='bar-table', figure=bar_table),
                html.Div([
                    html.H3("Attrition Bar Plot"),
                    html.P("In analysing attrition rates, there were 237 instances where employees answered 'yes' and 1233 instances where they responded 'no.' The 'yes' responses represent approximately 16.1% of the total count, while the 'no' responses constitute roughly 83.9%. "),
                    html.P("This data indicates that a relatively small percentage of individuals affirmed attrition compared to those who negated it. However, it's crucial to delve deeper into the reasons behind the affirmative responses to better understand the underlying factors contributing to attrition within the context studied.")
                ],style={"margin-top":"50px"}),
            ], style={'display': 'flex'}),

            html.Div([
                dcc.Graph(id='gender-pie-chart', figure=gender_pie_fig),
                dcc.Graph(id='gender-table', figure=gender_table),
                html.Div([
                    html.H3("Gender Pie Chart"),
                    html.P("The data regarding gender distribution shows that there were 882 males and 588 females. Males constitute approximately 60% of the total count, while females make up about 40%. This indicates a notable numerical advantage in male representation compared to females within the dataset.  "),
                    html.P("Analysing the percentages demonstrates a noticeable skew toward male individuals in the studied population. Further exploration into the specific contexts or domains where these gender disparities exist may provide valuable insights into potential areas for targeted interventions or initiatives aimed at promoting gender balance.")
                ],style={"margin-top":"50px"}),
            ], style={'display':'flex'}),
            html.Div([
                dcc.Graph(id='marital-status-pie-chart', figure=marital_status_pie_fig),
                dcc.Graph(id='marital-table', figure=marital_table),
                html.Div([
                    html.H3("Marital Status Pie chart"),
                    html.P("The dataset indicates that among the individuals surveyed, there were 673 respondents with a stated marital status and 237 who identified as divorced, while 470 identified as single. The percentage breakdown reveals that individuals with a specified marital status account for approximately 71.1% of the total count. "),
                    html.P("Among these, the divorced group represents around 35.2%, while singles constitute approximately 27.7%. This data signifies that a substantial majority of respondents provided information regarding their marital status, with a notable proportion being divorced individuals, highlighting the importance of considering their perspectives and circumstances in any comprehensive analysis or intervention strategies")
                ],style={"margin-top":"50px"}),
            ], style={'display': 'flex'}),
            html.Div([
                dcc.Graph(id='business-travel-bar-chart', figure=business_travel_bar_fig),
                dcc.Graph(id='busi-table', figure=busi_table),
                html.Div([
                    html.H3("Business Travel Bar Plot"),
                    html.P("The data on travel frequency reveals interesting insights about the surveyed individuals. Among them, 1043 respondents indicated they engage in regular travel, representing approximately 71.6% of the total count. Conversely, a smaller subset, consisting of 277 individuals, reported traveling frequently, constituting around 19% of the surveyed population. "),
                    html.P("Notably, the remaining 150 respondents mentioned that they do not travel, making up about 10.3% of the datasetThis breakdown underscores that a considerable majority, comprising over 90% of the surveyed group, engage in some form of travel, either regularly or frequently.")
                ],style={"margin-top":"50px"}),
            ], style={'display': 'flex'}),
            html.Div([
                dcc.Graph(id='education-field-bar-chart', figure=education_field_bar_fig),
                dcc.Graph(id='edu-table', figure=edu_table),
                html.Div([
                    html.H3("Education Field Bar Plot"),
                    html.P("The distribution of educational fields among the surveyed individuals presents a varied landscape. Among them, life sciences emerged as the predominant field, with 606 respondents indicating their educational background in this domain, constituting approximately 33.8% of the total count. Medical studies followed closely, with 464 individuals having pursued education in this field, accounting for around 25.9% of the surveyed population."),
                    html.P("In addition, 159 respondents reported having an educational background in marketing (about 8.9%), while 152 individuals mentioned possessing a technical degree (approximately 8.5%). Furthermore, a smaller subset, comprising 82 individuals, indicated their education falling under the category of 'other' (about 4.6%), which could encompass diverse fields not explicitly mentioned in the provided options. Finally, human resources stood as the least represented educational field among the respondents, with only 27 individuals (approximately 1.5%) indicating this as their educational background.This diverse distribution across multiple educational fields highlights the varied backgrounds of the surveyed individuals."),
                ],style={"margin-top":"20px"}),
            ], style={'display': 'flex'}),

            html.Div([
                dcc.Graph(id='department-pie-chart', figure=department_pie_fig),
                dcc.Graph(id='dept-table', figure=dept_table),
                html.Div([
                    html.H3("Department Pie Chart"),
                    html.P("The respondents, the research and development (R&D) department emerged as the most prevalent, with 961 individuals indicating their affiliation with this sector, constituting a substantial 60.6% of the total count. Sales came next, with 446 respondents aligning themselves with this department, accounting for approximately 28.2% of the surveyed population."),
                    html.P("Comparatively, the human resources (HR) department showed the smallest representation, with only 63 individuals reporting their association with this sector, making up roughly 4% of the total count. This data highlights a notable disparity in the distribution of individuals across these departments, signalling the dominance of the research and development sector within the surveyed group.")
                ],style={"margin-top":"50px"}),
            ], style={'display': 'flex'}),
            html.Div([
                dcc.Graph(id='overtime-pie-chart', figure=overtime_pie_fig),
                dcc.Graph(id='over-table', figure=over_table),
                html.Div([
                    html.H3("Overtime Pie Chart"),
                    html.P("Among the surveyed employees regarding overtime, 1054 individuals indicated they did not engage in overtime work, constituting approximately 71.7% of the total count. Conversely, 416 respondents reported participating in overtime activities, making up about 28.3% of the surveyed population. This data reveals that a considerable majority, around three-quarters of the surveyed employees, do not partake in overtime work."),
                ],style={"margin-top":"50px"}),
            ], style={'display': 'flex'}),
        ]),
        dcc.Tab(label='Box Plot - Age by Attrition', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Age by Attrition")),
                    dcc.Graph(id='graph-1', figure=fig1),
                ], className='six columns', style={'width': '40%'}),
                html.Div([
                    html.P(html.B("Employees who did not experience attrition ('No') had a count of 1233, with an average age of approximately 37.56 years. The age distribution had a standard deviation of around 8.89 years. The ages ranged from a minimum of 18 years to a maximum of 60 years. The interquartile range (IQR), which represents the middle 50% of the data, spanned from 31 years (25th percentile) to 43 years (75th percentile), with the median age (50th percentile) being 36 years.")),
                    html.P(html.B("On the other hand, for employees who experienced attrition ('Yes'), there were 237 data points available. The average age among this group was about 33.61 years, with a slightly higher standard deviation of approximately 9.69 years. The age range was similar, spanning from 18 years to 58 years. The IQR for this group ranged from 28 years (25th percentile) to 39 years (75th percentile), with the median age being 32 years.")),
                ], className='six columns', style={'width': '40%', 'margin-left': '100px', 'margin-top': '250px'}),
            ], className='row', style={'display': 'flex'}),
        ]),
        dcc.Tab(label='Box Plot - Monthly Income by Job Role and Attrition', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Monthly Income by Job Role and Attrition")),
                    dcc.Graph(id='graph-2', figure=fig2),
                ], style={'width':"50%"}),
                html.Div([
                    html.Table(style={'border': '2px solid', 'width': '100%', 'border-collapse': 'collapse'},className='table table-striped',children=[
                        # Table header
                        html.Tr([
                            html.Th("Job Role", colSpan=1),  
                            html.Th("No Attrition", colSpan=3),
                            html.Th("Attrition", colSpan=3),
                        ]),
                        html.Tr([
                            html.Th(""),
                            html.Th("Mean"),
                            html.Th("SD"),
                            html.Th("Range"),
                            html.Th("Mean"),
                            html.Th("SD"),
                            html.Th("Range"),
                        ]),
                        # Table data for each job role
                        html.Tr([
                            html.Td('Healthcare Representative'),
                            html.Td("$7453"),
                            html.Td("$2560"),
                            html.Td("$4000 - $13966"),
                            html.Td("$8548"),
                            html.Td("$2152"),
                            html.Td("$4777 - $12169"),
                        ]),
                        html.Tr([
                            html.Td('Human Resources'),
                            html.Td("$4391"),
                            html.Td("$2241"),
                            html.Td("$2064 - $10725"),
                            html.Td("$3715"),
                            html.Td("$3063"),
                            html.Td("$1555 - $10482"),
                        ]),
                        html.Tr([
                            html.Td('Laboratory Technician'),
                            html.Td("$3337"),
                            html.Td("$1172"),
                            html.Td("$1129 - $7403"),
                            html.Td("$2919"),
                            html.Td("$1019"),
                            html.Td("$1102 - $6074"),
                        ]),
                        html.Tr([
                            html.Td('Manager'),
                            html.Td("$17201"),
                            html.Td("$2245"),
                            html.Td("$11244 - $19999"),
                            html.Td("$16797"),
                            html.Td("$3788"),
                            html.Td("$11849 - $19859"),
                        ]),
                        html.Tr([
                            html.Td('Manufacturing Director'),
                            html.Td("$7289"),
                            html.Td("$2688"),
                            html.Td("$4011 - $13973"),
                            html.Td("$7365"),
                            html.Td("$2641"),
                            html.Td("$4171 - $10650"),
                        ]),
                        html.Tr([
                            html.Td('Research Director'),
                            html.Td("$15947"),
                            html.Td("$2810"),
                            html.Td("$11031 - $19973"),
                            html.Td("$19395"),
                            html.Td("$211"),
                            html.Td("$19246 - $19545"),
                        ]),
                        html.Tr([
                            html.Td('Research Scientist'),
                            html.Td("$3328"),
                            html.Td("$1230"),
                            html.Td("$1051 - $9724"),
                            html.Td("$2780"),
                            html.Td("$892"),
                            html.Td("$1009 - $4963"),
                        ]),
                        html.Tr([
                            html.Td('Sales Executive'),
                            html.Td("$6804"),
                            html.Td("$2301"),
                            html.Td("$4001 - $13872"),
                            html.Td("$7489"),
                            html.Td("$2602"),
                            html.Td("$4233 - $13758"),
                        ]),
                        html.Tr([
                            html.Td('Sales Representative'),
                            html.Td("$2798"),
                            html.Td("$900"),
                            html.Td("$1052 - $6632"),
                            html.Td("$2364"),
                            html.Td("$715"),
                            html.Td("$1081 - $4400"),
                        ]),
                    ]),
                ],style={"margin-top":"80px"})

            ],style={'display':'flex'})
        ]),
        dcc.Tab(label='Histogram - Monthly Income by Attrition', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Monthly Income by Attrition")),
                    dcc.Graph(id='graph-3', figure=fig3),
                ],style={'width':'40%'}),
                html.Div([
                    html.H3("The provided summary data illustrates the distribution of Monthly Income concerning Attrition status ('No' and 'Yes'). Here's the breakdown of the statistical values:"),
                    html.Div([
                        html.Div([
                            html.P(html.B("For employees who did not experience Attrition ('No'):")),
                            html.P("Count: 1233 individuals were considered."),
                            html.P("Mean: The average Monthly Income was approximately $6832."),
                            html.P("Standard Deviation: The dispersion of incomes around the mean was about $4818."),
                            html.P("Minimum: The lowest Monthly Income recorded was $1051."),
                            html.P("25th Percentile (Q1): 25% of employees had a Monthly Income below $3211."),
                            html.P("Median (50th Percentile): The median Monthly Income was $5204."),
                            html.P("75th Percentile (Q3): 75% of employees had a Monthly Income below $8834."),
                            html.P("Maximum: The highest Monthly Income recorded among employees was $19999."),
                        ]),
                        html.Div([
                            html.P(html.B("For employees who experienced Attrition ('Yes'):")),
                            html.P("Count: 237 individuals were considered."),
                            html.P("Mean: The average Monthly Income was approximately $4787."),
                            html.P("Standard Deviation: The dispersion of incomes around the mean was about $3640."),
                            html.P("Minimum: The lowest Monthly Income recorded was $1009."),
                            html.P("25th Percentile (Q1): 25% of employees had a Monthly Income below $2373."),
                            html.P("Median (50th Percentile): The median Monthly Income was $3202."),
                            html.P("75th Percentile (Q3): 75% of employees had a Monthly Income below $5916."),
                            html.P("Maximum: The highest Monthly Income recorded among employees was $19859."),
                        ])
                    ],style={"display":"flex"}),
                    html.P("These statistics provide insights into the distribution and central tendencies of Monthly Income among employees based on their Attrition status."),
                ],style={'width':'60%',"margin-left":"100px"}),
            ],style={"display":"flex"}),
        ]),
        dcc.Tab(label='Scatter Matrix Plot', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Scatter Matrix Plot")),
                    dcc.Graph(id='graph-4', figure=fig4),
                ],style={'width':'50%'}),
                html.Div([
                    html.P(html.H3("Job Level:")),
                    html.P("Employees who remained ('No' Attrition) had a higher mean job level (mean=2.15) compared to those who left ('Yes' Attrition) with a lower mean job level (mean=1.64)."),
                    html.P(html.H3("Total Working Years:")),
                    html.P("Employees who stayed ('No' Attrition) had a higher mean total working years (mean=11.86) compared to those who left ('Yes' Attrition) with a lower mean total working years (mean=8.24)."),
                    html.P(html.H3("Monthly Income:")),
                    html.P("Employees who remained ('No' Attrition) had a higher mean monthly income (mean=$6832) compared to those who left ('Yes' Attrition) with a lower mean monthly income (mean=$4787)."),
                    html.P(html.H3("Years in Current Role:")),
                    html.P("Employees who remained ('No' Attrition) had a higher mean years in the current role (mean=4.48) compared to those who left ('Yes' Attrition) with a lower mean years in the current role (mean=2.90)."),
                    html.P(html.H3("Performance Rating:")),
                    html.P("Both groups had similar performance ratings with a mean of around 3.15, regardless of Attrition status."),
                ],style={"width":"50%","margin-top":"150px"}),
            ],style={"display":"flex"}),
        ]),
        dcc.Tab(label='Bar Chart - Attrition by Job Role and Education Field', children=[
            html.Div([
                html.Center(html.H1("Attrition by Job Role and Education Field")),
                dcc.Graph(id='graph-5', figure=fig5),
            ]),
            html.Div([
                html.H3("Attrition by Job Role and Education Field:"),
                html.P("	Data presents counts, means, standard deviations, minimum and maximum values, and quartiles for different job roles categorized by educational fields concerning attrition ('Yes' or 'No'). This information illustrates the distribution of employees across these categories."),
                html.P("Within each job role, bars can be segmented based on educational fields ('Life Sciences,' 'Medical,' 'Other,' 'Technical Degree,' etc.). The height of each bar would represent the count of employees for the specific combination of job role, educational field, and attrition status."),
                html.P("Healthcare Representative: It shows a count of 58 employees with a 'Life Sciences' educational background and no attrition, while only 2 employees with 'Life Sciences' faced attrition. Similarly, there is one employee each with 'Medical' and 'Other' educational backgrounds facing attrition in this role."),
                html.B("Human Resources: "),html.P("There are 14 employees with a 'Human Resources' educational background and no attrition, and 7 employees facing attrition in the same field of study."),
                html.B("Laboratory Technician: "),html.P("86 employees with 'Life Sciences' and no attrition, and 33 employees with 'Life Sciences' experiencing attrition. Likewise, 82 employees with 'Medical' and no attrition, with no further data for attrition status in the 'Medical' field for this role."),
            ])
        ]),
        dcc.Tab(label='Box Plot - Business Travel by Marital Status and Attrition', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Business Travel by Marital Status and Attrition")),
                    dcc.Graph(id='graph-6', figure=fig6),
                ],),
                html.Div([
                    html.P(html.H3("Non-Travel:")),
                    html.P("Among Divorced employees, 43 did not experience Attrition ('No') and 1 had Attrition ('Yes')."),
                    html.P("For Married individuals, 56 did not experience Attrition and 3 had Attrition."),
                    html.P("In the Single category, 39 did not experience Attrition and 8 had Attrition."),
                    html.P(html.H3("Travel-Frequently:")),
                    html.P("Among Divorced employees, 50 did not experience Attrition and 13 had Attrition."),
                    html.P("For Married individuals, 99 did not experience Attrition and 19 had Attrition."),
                    html.P("In the Single category, 59 did not experience Attrition and 37 had Attrition."),
                    html.P(html.H3("Travel-Rarly:")),
                    html.P("Among Divorced employees, 201 did not experience Attrition and 19 had Attrition."),
                    html.P("For Married individuals, 434 did not experience Attrition and 62 had Attrition."),
                    html.P("In the Single category, 252 did not experience Attrition and 75 had Attrition."),
                ],style={"margin-left":"20px","margin-top":"50px"}),
            ],style={"display":"flex"}),
        ]),
        dcc.Tab(label='Scatter Matrix Plot - Monthly Income, Hourly Rate, Distance from Home', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Monthly Income, Hourly Rate, Distance from Home")),
                    dcc.Graph(id='graph-7', figure=fig7)
                ],style={'width':'50%'}),
                html.Div([
                    html.P(html.H3("Monthly Income:")),
                    html.P("Employees who didn't experience Attrition ('No') had a notably higher average monthly income (mean=$6832) compared to those who left ('Yes' Attrition), where the mean income was lower (mean=$4787)."),
                    html.P("The range between the minimum and maximum monthly income for both groups ('No' and 'Yes' Attrition) was substantial, indicating diverse income distributions among employees."),
                    html.P(html.H3("Hourly Rate:")),
                    html.P("The average hourly rate for employees who stayed ('No' Attrition) was slightly higher (mean=approximately $66) compared to those who left ('Yes' Attrition) with a slightly lower average rate (mean=approximately $65)."),
                    html.P("The minimum and maximum hourly rates for both groups were the same (ranging from $30 to $100), showing consistency in this regard."),
                    html.P(html.H3("Distance from Home:")),
                    html.P("Employees who left ('Yes' Attrition) had a slightly higher average distance from home (mean=approximately 10.63 miles) compared to those who stayed ('No' Attrition) with a lower average distance (mean=approximately 8.92 miles)."),
                    html.P("The range and quartile values indicate a broader dispersion of distances among employees who experienced Attrition ('Yes') compared to those who did not."),
                ],style={"width":"50%","margin-left":"50px","margin-top":"100px"}),
            ],style={"display":"flex"}),
        ]),
        dcc.Tab(label='HeatMap - Correlation', children=[
            html.Div([
                html.Div([
                    html.Center(html.H1("Correlation")),
                    dcc.Graph(id='graph-8', figure=heatmap_fig)
                ],style={"width":"40%"}),
                html.Div([
                    html.H3("Correlation Map:"),
                    html.P("The correlation map provided indicates the relationships between various attributes within the dataset. Each cell in the map displays the correlation coefficient between two attributes, ranging from -1 to 1. A correlation of 1 implies a perfect positive correlation, -1 denotes a perfect negative correlation, and 0 indicates no correlation."),
                    html.B("Analyzing the correlation map:"),
                    html.P("Age and JobLevel exhibit a strong positive correlation of around 0.51, indicating that as the age of employees increases, their job level tends to rise as well."),
                    html.P("JobLevel and MonthlyIncome show a notably strong positive correlation of about 0.95, suggesting that higher job levels are associated with higher monthly incomes, which is expected."),
                    html.P("TotalWorkingYears and YearsAtCompany display a strong positive correlation of approximately 0.63, implying that the total number of years an employee has worked is closely related to the duration spent at their current company."),
                    html.P("YearsInCurrentRole and YearsWithCurrManager have a relatively high positive correlation of around 0.76, indicating that the time an employee spends in their current role is closely linked to the duration they have been with their current manager."),
                    html.P("Conversely, attributes like PerformanceRating and EnvironmentSatisfaction show weaker correlations closer to zero, suggesting a lack of significant linear relationship between these factors")
                ],style={"width":"60%","margin-left":"50px","margin-top":"200px"}),
            ],style={"display":"flex"}),
        ]),
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
