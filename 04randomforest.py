# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:33:39 2019

@author: Amartya
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#%%
data1 = pd.read_csv("./dataset/accidents_2005_to_2007.csv")
data2 = pd.read_csv("./dataset/accidents_2009_to_2011.csv")
data3 = pd.read_csv("./dataset/accidents_2012_to_2014.csv")
#%%
data = pd.concat([data1, data2, data3])
#%%
data['date_time'] = data['Date']+ ' ' + data['Time']
time_format = '%d/%m/%Y %H:%M'
data['date_time'] = pd.to_datetime(data['date_time'], format=time_format)
data['Month'] = data['date_time'].dt.month
data['Hour'] = data['date_time'].dt.hour
data['Day'] = data['date_time'].dt.day
#%%
data.drop(['date_time','Date','Time','Location_Easting_OSGR', 'Location_Northing_OSGR',
       '1st_Road_Class','1st_Road_Number','2nd_Road_Class',
       '2nd_Road_Number','Accident_Index','Police_Force','Junction_Detail',
       'Did_Police_Officer_Attend_Scene_of_Accident','LSOA_of_Accident_Location'], axis=1, inplace=True)
#%%
data.dropna(inplace=True)

data = data[data.Weather_Conditions!='Unknown']
data = data[data.Road_Type!='Unknown']
#%%
le = LabelEncoder()
data["Pedestrian_Crossing-Physical_Facilities"]= le.fit_transform(data["Pedestrian_Crossing-Physical_Facilities"])
data["Light_Conditions"]= le.fit_transform(data["Light_Conditions"])
data["Weather_Conditions"] = le.fit_transform(data["Weather_Conditions"])
data["Road_Surface_Conditions"] = le.fit_transform(data["Road_Surface_Conditions"])
data["Pedestrian_Crossing-Human_Control"] = le.fit_transform(data["Pedestrian_Crossing-Human_Control"])
data["Road_Type"] = le.fit_transform(data["Road_Type"])
data["Junction_Control"] = le.fit_transform(data["Junction_Control"])
data["Carriageway_Hazards"] = le.fit_transform(data["Carriageway_Hazards"])
data["Special_Conditions_at_Site"] = le.fit_transform(data["Special_Conditions_at_Site"])
data["Local_Authority_(Highway)"] = le.fit_transform(data["Local_Authority_(Highway)"])
data["Local_Authority_(District)"] = le.fit_transform(data["Local_Authority_(District)"])
#%%
onehot = pd.get_dummies(data.Accident_Severity,prefix=['Severity'])
data["Not_slight"] = onehot["['Severity']_1"] + onehot["['Severity']_2"]
data["Slight"] = onehot["['Severity']_3"]
#%%
data.drop(['Accident_Severity'], axis=1, inplace=True)
#%%
data = data.fillna(0)
#%%
features = ['Longitude',
       'Latitude', 'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week',
       'Road_Type', 'Speed_limit', 'Pedestrian_Crossing-Human_Control',
       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
       'Year', 'Month', 'Hour', 'Day','Junction_Control',
       'Local_Authority_(Highway)', 'Local_Authority_(District)','Carriageway_Hazards',
       'Special_Conditions_at_Site']
labels = ['Slight','Not_slight']
#%%
RANDOM_SEED = 314 #used to help randomly select the data points
TEST_PCT = 0.2
#%%
train_x, test_x = train_test_split(data, test_size=TEST_PCT, random_state=RANDOM_SEED)
train_y = train_x[labels]
test_y = test_x[labels]
train_x.drop(labels, axis=1, inplace=True)
test_x.drop(labels, axis=1, inplace=True)
#%%
model = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=4)
#%%
model.fit(train_x , train_y)
#%%
y_pred = model.predict(test_x)
print(classification_report(test_y, y_pred,target_names=labels))

