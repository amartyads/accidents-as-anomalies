# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:47:03 2019

@author: Amartya
"""

import pandas as pd
import numpy as np


from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
#%%
from math import pi
from bokeh.palettes import Category20c,viridis
from bokeh.transform import cumsum
#%%
accisev = ['','Fatal','Severe','Slight']
#%%
def piechart(src,heads,val,acci,title=''):
    title = title+accisev[acci] + " casualties, "+ val+" by "+heads
    src.columns = src.columns.get_level_values(0)
    src['angle'] = src[val]/src[val].sum() * 2*pi
    src['percent'] = src[val]/src[val].sum() * 100.00
    src['color'] = viridis(len(src.index))


    p = figure(plot_height=350, title=title, toolbar_location=None,
               tools="hover", tooltips="@"+heads+": @"+val + ", @percent" , x_range=(-0.5, 1.0))
    
    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend=heads, source=src)
    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    
    show(p)
    
def barchart(src,heads,val,acci,title=''):
    title = title+accisev[acci] + " casualties, "+ val+" by "+heads
    src.columns = src.columns.get_level_values(0)
    colorpalette = viridis(len(src.index))
    src['color'] = colorpalette
    p = figure(title=title,
             plot_height=400, plot_width=700, tools="hover", tooltips="@"+heads+": @"+val,
             x_axis_label=heads, y_axis_label='Values',toolbar_location=None,
             x_minor_ticks=2, y_range=(0, src[val].max() + (src[val].max()//10)))
    p.vbar(x=heads, bottom=0, top=val,
         color='color', width=0.75,source=src)
    show(p)
    

#%%
output_file('./trafficvis.html')
#%%
data1 = pd.read_csv("./dataset/accidents_2005_to_2007.csv")
data2 = pd.read_csv("./dataset/accidents_2009_to_2011.csv")
data3 = pd.read_csv("./dataset/accidents_2012_to_2014.csv")
#%%
data = pd.concat([data1, data2, data3])
#%%
cas_table = data.groupby(['Day_of_Week']).agg({'Number_of_Casualties':['sum']})
#%%
novehic_table = cas_table = data.groupby(['Day_of_Week']).agg({'Number_of_Vehicles':['min','max']})
#%%

fatal_table = data[data.Accident_Severity == 1].groupby(['Day_of_Week']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Day_of_Week'})
severe_table = data[data.Accident_Severity == 2].groupby(['Day_of_Week']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Day_of_Week'})
slight_table = data[data.Accident_Severity == 3].groupby(['Day_of_Week']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Day_of_Week'})
#%%
barchart(fatal_table,'Day_of_Week','Number_of_Casualties',1)
#%%
barchart(severe_table,'Day_of_Week','Number_of_Casualties',2)
#%%
barchart(slight_table,'Day_of_Week','Number_of_Casualties',3)
#%%
fatal_light_table = data[data.Accident_Severity == 1].groupby(['Light_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Lighting_Conditions'})
severe_light_table = data[data.Accident_Severity == 2].groupby(['Light_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Lighting_Conditions'})
slight_light_table = data[data.Accident_Severity == 3].groupby(['Light_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Lighting_Conditions'})
#%%
piechart(fatal_light_table, 'Light_Conditions','Number_of_Casualties',1)
#%%
piechart(severe_light_table, 'Light_Conditions','Number_of_Casualties',2)
#%%
piechart(slight_light_table, 'Light_Conditions','Number_of_Casualties',3)
#%%
fatal_weather_table = data[data.Accident_Severity == 1].groupby(['Weather_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Weather_Conditions'})
piechart(fatal_weather_table,'Weather_Conditions','Number_of_Casualties',1)
#%%
severe_weather_table = data[data.Accident_Severity == 2].groupby(['Weather_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Weather_Conditions'})
piechart(severe_weather_table,'Weather_Conditions','Number_of_Casualties',2)
#%%
slight_weather_table = data[data.Accident_Severity == 3].groupby(['Weather_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Weather_Conditions'})
piechart(slight_weather_table,'Weather_Conditions','Number_of_Casualties',3)
#%%
fatal_road_table = data[data.Accident_Severity == 1].groupby(['Road_Surface_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Road_Surface_Conditions'})
piechart(fatal_road_table,'Road_Surface_Conditions','Number_of_Casualties',1)
#%%
severe_road_table = data[data.Accident_Severity == 2].groupby(['Road_Surface_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Road_Surface_Conditions'})
piechart(severe_road_table,'Road_Surface_Conditions','Number_of_Casualties',2)
#%%
slight_road_table = data[data.Accident_Severity == 3].groupby(['Road_Surface_Conditions']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Road_Surface_Conditions'})
piechart(slight_road_table,'Road_Surface_Conditions','Number_of_Casualties',3)
#%%
data['Urban_or_Rural_Area'] = data['Urban_or_Rural_Area'].map({1:'Urban', 2:'Rural',3: 'Unknown'})
fatal_urbrur_table = data[data.Accident_Severity == 1].groupby(['Urban_or_Rural_Area']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Urban_or_Rural_Area'})
piechart(fatal_urbrur_table,'Urban_or_Rural_Area','Number_of_Casualties',1)
#%%
severe_urbrur_table = data[data.Accident_Severity == 2].groupby(['Urban_or_Rural_Area']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Urban_or_Rural_Area'})
piechart(severe_urbrur_table,'Urban_or_Rural_Area','Number_of_Casualties',2)
#%%
slight_urbrur_table = data[data.Accident_Severity == 3].groupby(['Urban_or_Rural_Area']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Urban_or_Rural_Area'})
piechart(slight_urbrur_table,'Urban_or_Rural_Area','Number_of_Casualties',3)
#%%
data['date_time'] = data['Date']+ ' ' + data['Time']
time_format = '%d/%m/%Y %H:%M'
data['date_time'] = pd.to_datetime(data['date_time'], format=time_format)
#%%
data['Month'] = data['date_time'].dt.month
data['Hour'] = data['date_time'].dt.hour
#%%
fatal_month_table = data[data.Accident_Severity == 1].groupby(['Month']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Month'})
barchart(fatal_month_table,'Month','Number_of_Casualties',1)
#%%
severe_month_table = data[data.Accident_Severity == 2].groupby(['Month']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Month'})
barchart(severe_month_table,'Month','Number_of_Casualties',2)
#%%
slight_month_table = data[data.Accident_Severity == 3].groupby(['Month']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Month'})
barchart(slight_month_table,'Month','Number_of_Casualties',3)
#%%
fatal_hour_table = data[data.Accident_Severity == 1].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(fatal_hour_table,'Hour','Number_of_Casualties')
#%%
severe_hour_table = data[data.Accident_Severity == 2].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(severe_hour_table,'Hour','Number_of_Casualties')
#%%
slight_hour_table = data[data.Accident_Severity == 3].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(slight_hour_table,'Hour','Number_of_Casualties')

#%%
weekend_data = data[data['Day_of_Week'].isin([1,7])]
weekday_data = data[data['Day_of_Week'].isin([2,3,4,5,6])]
#%%
fatalwd_hour_table = weekday_data[weekday_data.Accident_Severity == 1].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(fatalwd_hour_table,'Hour','Number_of_Casualties',1,"Weekday ")
#%%
severewd_hour_table = weekday_data[weekday_data.Accident_Severity == 2].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(severewd_hour_table,'Hour','Number_of_Casualties',2,"Weekday ")
#%%
slightwd_hour_table = weekday_data[weekday_data.Accident_Severity == 3].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(slightwd_hour_table,'Hour','Number_of_Casualties',3,"Weekday ")
#%%
fatalwn_hour_table = weekend_data[weekend_data.Accident_Severity == 1].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(fatalwn_hour_table,'Hour','Number_of_Casualties',1,"Weekend ")
#%%
severewn_hour_table = weekend_data[weekend_data.Accident_Severity == 2].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(severewn_hour_table,'Hour','Number_of_Casualties',2,"Weekend ")
#%%
slightwn_hour_table = weekend_data[weekend_data.Accident_Severity == 3].groupby(['Hour']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Hour'})
barchart(slightwn_hour_table,'Hour','Number_of_Casualties',3,"Weekend ")
#%%
fatalwd_month_table = weekday_data[weekday_data.Accident_Severity == 1].groupby(['Month']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Month'})
barchart(fatalwd_month_table,'Month','Number_of_Casualties')
#%%
fatalwn_month_table = weekend_data[weekend_data.Accident_Severity == 1].groupby(['Month']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Month'})
barchart(fatalwn_month_table,'Month','Number_of_Casualties')
#%%
fatal_splm_table = data[data.Accident_Severity == 1].groupby(['Speed_limit']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Speed_limit'})
piechart(fatal_splm_table,'Speed_limit','Number_of_Casualties',1)
#%%
severe_splm_table = data[data.Accident_Severity == 2].groupby(['Speed_limit']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Speed_limit'})
piechart(severe_splm_table,'Speed_limit','Number_of_Casualties',2)
#%%
slight_splm_table = data[data.Accident_Severity == 3].groupby(['Speed_limit']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Speed_limit'})
piechart(slight_splm_table,'Speed_limit','Number_of_Casualties',3)
#%%
fatal_roadt_table = data[data.Accident_Severity == 1].groupby(['Road_Type']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Road_Type'})
piechart(fatal_roadt_table,'Road_Type','Number_of_Casualties',1)
#%%
severe_roadt_table = data[data.Accident_Severity == 2].groupby(['Road_Type']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Road_Type'})
piechart(severe_roadt_table,'Road_Type','Number_of_Casualties',2)
#%%
slight_roadt_table = data[data.Accident_Severity == 3].groupby(['Road_Type']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Road_Type'})
piechart(slight_roadt_table,'Road_Type','Number_of_Casualties',3)
#%%
fatal_splmrd_table = data[data.Accident_Severity == 1].groupby(['Speed_limit','Road_Surface_Conditions']).agg({'Number_of_Casualties':['sum']})
fatal_splmrd_table = fatal_splmrd_table.reset_index()
#%%
fatal_cross_table = data[data.Accident_Severity == 1].groupby(['Pedestrian_Crossing-Physical_Facilities']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Pedestrian_Crossing-Physical_Facilities'})
piechart(fatal_cross_table,'Pedestrian_Crossing-Physical_Facilities','Number_of_Casualties',1)
#%%
severe_cross_table = data[data.Accident_Severity == 2].groupby(['Pedestrian_Crossing-Physical_Facilities']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Pedestrian_Crossing-Physical_Facilities'})
piechart(severe_cross_table,'Pedestrian_Crossing-Physical_Facilities','Number_of_Casualties',2)
#%%
slight_cross_table = data[data.Accident_Severity == 3].groupby(['Pedestrian_Crossing-Physical_Facilities']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Pedestrian_Crossing-Physical_Facilities'})
piechart(slight_cross_table,'Pedestrian_Crossing-Physical_Facilities','Number_of_Casualties',3)