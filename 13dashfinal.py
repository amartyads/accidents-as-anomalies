# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:27:07 2019

@author: Amartya
"""

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
#%%
accisev = ['','Fatal','Severe','Slight']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = [r'C:\Users\Amartya\TUM\master\carta-master\src\scss\main.scss']
#%%
def prepfataldata(data, grouper):
    table1 = data[data.Accident_Severity == 1].groupby([grouper]).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':grouper})
    table1.columns = table1.columns.get_level_values(0)
    return table1
def prepseveredata(data, grouper):
    table2 = data[data.Accident_Severity == 2].groupby([grouper]).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':grouper})
    table2.columns = table2.columns.get_level_values(0)
    return table2
def prepslightdata(data, grouper):
    table3 = data[data.Accident_Severity == 3].groupby([grouper]).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':grouper})
    table3.columns = table3.columns.get_level_values(0)
    return table3
#%%
def bargraph(curData):
    toret = {
            'data':[go.Bar(
                    x=curData[curData.columns[0]].tolist(), 
                    y=curData[curData.columns[1]].tolist(),
                    marker={
                        'color': curData[curData.columns[1]].tolist(),
                        'colorscale': 'Viridis'
                    })],
            'layout': {}
            }
    return toret
#%%
def piechart(curData):
    toret = {
            'data':[go.Pie(
                    labels=curData[curData.columns[0]].tolist(), 
                    values=curData[curData.columns[1]].tolist())],
            'layout': go.Layout(showlegend=False)
            }
    return toret
#%%
bars = ['Day_of_Week','Month','Hour','Year','Number_of_Vehicles']
pies = ['Weather_Conditions','Light_Conditions','Road_Surface_Conditions','Urban_or_Rural_Area','Speed_limit','Road_Type','Pedestrian_Crossing-Physical_Facilities']
datachunk = ['All days','Weekdays', 'Weekends']
def getfig(curData):
    
    if curData.columns[0] in bars:
        return bargraph(curData)
    else:
        return piechart(curData)
#%%
data = pd.DataFrame()
weekday_data = pd.DataFrame()
weekend_data = pd.DataFrame()
def retData(datastr):
    if datastr == 'All days':
        return data
    elif datastr == 'Weekdays':
        return weekday_data
    else:
        return weekend_data
#%%
#cd C:\Users\Amartya\TUM\99 Projects\75 Accident prediction
#%%
def prepdata():
    print("reading data")
    data1 = pd.read_csv("./dataset/accidents_2005_to_2007.csv")
    data2 = pd.read_csv("./dataset/accidents_2009_to_2011.csv")
    data3 = pd.read_csv("./dataset/accidents_2012_to_2014.csv")
    print("data read")
    #%%
    data = pd.concat([data1, data2, data3])
    #%%
    data['date_time'] = data['Date']+ ' ' + data['Time']
    time_format = '%d/%m/%Y %H:%M'
    data['date_time'] = pd.to_datetime(data['date_time'], format=time_format)
    #%%
    data['Month'] = data['date_time'].dt.month
    data['Hour'] = data['date_time'].dt.hour
    data['Urban_or_Rural_Area'] = data['Urban_or_Rural_Area'].map({1:'Urban', 2:'Rural',3: 'Unknown'})
    #%%
    weekend_data = data[data['Day_of_Week'].isin([1,7])]
    weekday_data = data[data['Day_of_Week'].isin([2,3,4,5,6])]
    return data, weekend_data, weekday_data





#%%
app = dash.Dash(__name__ ,external_stylesheets=external_stylesheets)
#dcc._css_dist[0]['relative_package_path'].append(r'C:\Users\Amartya\TUM\master\carta-master\src\scss\main.scss')
colors = {
    'background': '#FFFFFF',
    'text': '#FF00FF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Data Visualization'
    ),
    html.Div([
            dcc.Dropdown(id='accitype', options=[{'label':i, 'value':i} for i in (bars+pies)],value=bars[0]),
            ], style={'width':'20%','float':'left', 'display':'inline-block', 'marginRight':'1em'}),
    html.Div([
            dcc.Dropdown(id='daytype', options=[{'label':i, 'value':i} for i in datachunk],value=datachunk[0])
            ], style={'width':'20%', 'display':'inline-block', 'marginRight':'1em'}),
    html.Br(),
    html.Div([html.Div(children='Fatal casualties'),
            dcc.Graph(id='fatalgraph')], 
                style={'width':'33%','display':'inline-block','float':'left'}),
    html.Div([html.Div(children='Severe casualties'),
            dcc.Graph(id='severegraph')], 
                style={'width':'33%','display':'inline-block','float':'center'}),
    html.Div([html.Div(children='Slight casualties'),
            dcc.Graph(id='slightgraph')], 
                style={'width':'33%','display':'inline-block','float':'right'})
])
#%%
@app.callback(
        Output('fatalgraph','figure'),
        [Input('accitype','value'),
         Input('daytype','value')])
def update_fatalgraph(accitypeval, daytypeval):
    fatal = prepfataldata(data=retData(daytypeval), grouper=accitypeval)
    return getfig(fatal)
#%%
@app.callback(
        Output('severegraph','figure'),
        [Input('accitype','value'),
         Input('daytype','value')])
def update_severegraph(accitypeval, daytypeval):
    severe = prepseveredata(data=retData(daytypeval), grouper=accitypeval)
    return getfig(severe)
#%%
@app.callback(
        Output('slightgraph','figure'),
        [Input('accitype','value'),
         Input('daytype','value')])
def update_slightgraph(accitypeval, daytypeval):
    slight = prepslightdata(data=retData(daytypeval), grouper=accitypeval)
    return getfig(slight)
#%%
if __name__ == '__main__':
    data, weekend_data, weekday_data = prepdata()
    app.run_server()
    