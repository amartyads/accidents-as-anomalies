# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:24:54 2019

@author: Amartya
"""

import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
#%%
accisev = ['','Fatal','Severe','Slight']
#%%
def prepdata(grouper):
    table1 = data[data.Accident_Severity == 1].groupby([grouper]).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':grouper})
    table1.columns = table1.columns.get_level_values(0)
    table2 = data[data.Accident_Severity == 2].groupby([grouper]).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':grouper})
    table2.columns = table2.columns.get_level_values(0)
    table3 = data[data.Accident_Severity == 3].groupby([grouper]).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':grouper})
    table3.columns = table3.columns.get_level_values(0)
    return table1,table2,table3
#%%
def bargraph(curData, graphID = 'temp'):
    toret = dcc.Graph(
            id = graphID,
            figure = {
                    'data':[{'x':curData[curData.columns[0]].tolist(),'y':curData[curData.columns[1]].tolist(), 'type':'bar'}],
                    'layout':{}
                    }
            )
    return toret
#%%
def piechart(curData, graphID = 'temp'):
    toret = dcc.Graph(
            id = graphID,
            figure = {
                    'data':[go.Pie(labels=curData[curData.columns[0]].tolist(), values=curData[curData.columns[1]].tolist())],
                    'layout':go.Layout()
                    }
            )
    return toret
#%%
bars = ['Day_of_Week','Month','Hour']
pies = ['Weather_Conditions','Light_Conditions','Road_Surface_Conditions','Urban_or_Rural_Area','Speed_limit','Road_Type','Pedestrian_Crossing-Physical_Facilities']
def getfig(curData, graphID = 'temp'):
    if curData.columns[0] in bars:
        return bargraph(curData, graphID)
    else:
        return piechart(curData, graphID)
#%%
#cd C:\Users\Amartya\TUM\99 Projects\75 Accident prediction
#%%
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
#%%
fatal_week, severe_week, slight_week = prepdata('Day_of_Week')
#%%
fatal_weather, severe_weather, slight_weather = prepdata('Weather_Conditions')
#%%
app = dash.Dash()
colors = {
    'background': '#FFFFFF',
    'text': '#FF00FF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'font-family': 'Segoe UI',
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    getfig(slight_weather)
])
#%%
if __name__ == '__main__':
    app.run_server(debug=True)