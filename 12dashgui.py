# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:57:42 2019

@author: Amartya
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
#%%
bars = ['Day_of_Week','Month','Hour']
pies = ['Weather_Conditions','Light_Conditions','Road_Surface_Conditions','Urban_or_Rural_Area','Speed_limit','Road_Type','Pedestrian_Crossing-Physical_Facilities']
datachunk = ['All days','Weekdays', 'Weekends']
#%%
app = dash.Dash()
#%%
app.layout = html.Div([
    html.Div([
            dcc.Dropdown(id='accitype', options=[{'label':i, 'value':i} for i in (bars+pies)],value=bars[0]),
            ], style={'width':'20%','float':'left', 'display':'inline-block', 'marginRight':'1em'}),
    html.Div([
            dcc.Dropdown(id='daytype', options=[{'label':i, 'value':i} for i in datachunk],value=datachunk[0])
            ], style={'width':'20%', 'display':'inline-block', 'marginRight':'1em'}),
    html.Div(id='my-div')
])
#%%
@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='accitype', component_property='value'),
     Input('daytype','value')]
)
def update_output_div(input_value1, input_value2):
    return 'You\'ve entered "{}"'.format(input_value1+input_value2)
#%%
if __name__ == '__main__':
    app.run_server(debug=True)