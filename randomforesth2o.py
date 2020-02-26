# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:51:17 2019

@author: Amartya
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import h2o
from h2o.estimators import H2ORandomForestEstimator,  H2OGradientBoostingEstimator
from sklearn.preprocessing import LabelEncoder
#%%
cd C:\Users\Amartya\TUM\99 Projects\75 Accident prediction
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
data = data.fillna(0)
#%%
features = ['Longitude',
       'Latitude', 'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week',
       'Road_Type', 'Speed_limit', 'Pedestrian_Crossing-Human_Control',
       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
       'Year', 'Month', 'Hour', 'Day','Junction_Control',
       'Local_Authority_(Highway)', 'Local_Authority_(District)','Carriageway_Hazards',
       'Special_Conditions_at_Site',]
labels = 'Accident_Severity'
#%%
mask = (data.Accident_Severity < 3)
column_name = labels
data.loc[mask, column_name] = 1
#%%
datah1 = data[data.Accident_Severity == 1]
datah2 = data[data.Accident_Severity == 3].sample(n=343221,random_state=314)
datas = pd.concat([datah1,datah1,datah1,datah2])
datas = datas.sample(frac=1).reset_index(drop=True)
#%%
h2o.init(ip='localhost', nthreads=-1, min_mem_size='2G', max_mem_size='4G')
#%%
h2odata = h2o.H2OFrame(datas)
#%%
train, test = h2odata.split_frame(ratios=[0.8])
train[labels] = train[labels].asfactor()
test[labels] = test[labels].asfactor()
#%%
model = H2ORandomForestEstimator(ntrees=80, max_depth=20, nfolds=10)
#%%
model.train(x=features, y=labels, training_frame=train)
#%%
performance = model.model_performance(test_data=test)

print(performance.show())
#%%
my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                      ntrees=30,
                                      max_depth=5,
                                      min_rows=2,
                                      learn_rate=0.2,
                                      nfolds=5,
                                      fold_assignment="Modulo",
                                      keep_cross_validation_predictions=True,
                                      seed=1)
my_gbm.train(x=features, y=labels, training_frame=train)
#%%
performance = my_gbm.model_performance(test_data=test)

print(performance.show())