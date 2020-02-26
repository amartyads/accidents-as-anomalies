# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:40:36 2019

@author: Amartya
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#%%
#tf.enable_eager_execution()

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
data.drop(['date_time','Date','Time','1st_Road_Class','1st_Road_Number','2nd_Road_Class','2nd_Road_Number','Accident_Index','Police_Force', 'Junction_Detail', 'Junction_Control', 'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Did_Police_Officer_Attend_Scene_of_Accident', 'LSOA_of_Accident_Location', 'Local_Authority_(District)', 'Local_Authority_(Highway)'], axis=1, inplace=True)
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
#%%
onehot = pd.get_dummies(data.Accident_Severity,prefix=['Severity'])
data["Fatal"] = onehot["['Severity']_1"]
data["Severe"] = onehot["['Severity']_2"]
data["Slight"] = onehot["['Severity']_3"]
#%%
data.drop(['Accident_Severity'], axis=1, inplace=True)
data = (data-data.min())/(data.max()-data.min())
#%%
data = data.fillna(0)
#%%
features = ['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude',
       'Latitude', 'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week',
       'Road_Type', 'Speed_limit', 'Pedestrian_Crossing-Human_Control',
       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
       'Year', 'Month', 'Hour', 'Day']
labels = ['Slight','Fatal','Severe']
#%%
data['Longitude'] = round(data['Longitude'],1)
data['Latitude'] = round(data['Latitude'],1)
data['Location_Easting_OSGR'] = round(data['Location_Easting_OSGR'],1)
data['Location_Northing_OSGR'] = round(data['Location_Northing_OSGR'],1)
#%%
data.sort_values(by=['Year','Month','Day','Hour'],axis=0,inplace=True)
#%%
datah1 = data[data.Fatal == 1]
datah2 = data[data.Severe == 1].iloc[:40000]
datah3 = data[data.Slight == 1].iloc[:40000]
datas = pd.concat([datah1,datah1,datah2,datah3])
#%%
datah = datas.drop(labels,axis=1)
features_np = datah.values
labels_np = datas[labels].values
#%%
features_np = features_np.reshape(features_np.shape[0], features_np.shape[1],1)
#%%
num_data = labels_np.shape[0]
train_split  = 0.8
num_train = int(num_data * train_split)
num_test = num_data - num_train
indices = np.random.permutation(num_data)
#%%
train_split = (num_data * 80) // 100
train_idx, test_idx = indices[:train_split], indices[train_split:]

features_train = np.array([features_np[i] for i in train_idx])
features_test = np.array([features_np[i] for i in test_idx])

labels_train = np.array([labels_np[i] for i in train_idx])
labels_test = np.array([labels_np[i] for i in test_idx])
#%%
batch_size = 1024
train_dataset = tf.data.Dataset.from_tensor_slices((features_train,labels_train)).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((features_test,labels_test)).batch(batch_size).repeat()
#%%
model = keras.Sequential([
    keras.layers.GRU(units=16, input_shape=[19,1], return_sequences=True),
    keras.layers.GRU(units=8, return_sequences=True),
    keras.layers.GRU(units=4, return_sequences=False),
    keras.layers.Dense(units=3, activation='softmax')
])
#%%
model.compile('adamax','categorical_crossentropy', metrics=['accuracy'])
#%%
model.summary()
#%%
stepsno = (features_np.shape[0] // batch_size) + 1
history = model.fit(train_dataset,steps_per_epoch=stepsno, epochs=100)
#%%
predictions = model.predict(data.head().drop(labels,axis=1), steps=100)
#%%
test_loss, test_acc = model.evaluate(test_dataset,steps=24)

print('Test accuracy:', test_acc)
#%%
datah = data.drop(labels,axis=1)
features_np = datah.values
labels_np = data[labels].values
#%%
features_np = features_np.reshape(features_np.shape[0], features_np.shape[1],1)
#%%
test_loss, test_acc = model.evaluate(features_np,labels_np)

print('Test accuracy:', test_acc)