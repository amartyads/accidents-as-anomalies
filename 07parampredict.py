# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:54:41 2019

@author: Amartya
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#%%
def data_convert(dataset,timesteps,featNum=3):
    maxLim = dataset.shape[0]
    features = [[[None] * (featNum)] * timesteps] * (maxLim - timesteps)
    labels = [None] * (maxLim - timesteps)
    for i in range(0,maxLim - timesteps):
        features[i] = dataset[i:i+timesteps].values
        labels[i] = dataset['Number_of_Casualties'][(i+timesteps)]
    return np.array(features), np.array(labels)
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
#%%
data['Hour'] = data['date_time'].dt.hour
data['Day'] = data['date_time'].dt.day
#%%
data.drop(['date_time','Time','Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude',
       'Latitude', '1st_Road_Class','1st_Road_Number','2nd_Road_Class',
       '2nd_Road_Number','Accident_Index','Police_Force','Junction_Detail',
       'Did_Police_Officer_Attend_Scene_of_Accident'], axis=1, inplace=True)
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
data["LSOA_of_Accident_Location"] = le.fit_transform(data["LSOA_of_Accident_Location"])
#%%
onehot = pd.get_dummies(data.Accident_Severity,prefix=['Severity'])
data["Fatal"] = onehot["['Severity']_1"]
#%%
#data.drop(['Accident_Severity'], axis=1, inplace=True)
data['Date'] = data['Day'].astype(int).astype(str) + '/' + data['Month'].astype(int).astype(str) + '/' + data['Year'].astype(str)
#%%
features = ['Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week',
       'Road_Type', 'Speed_limit', 'Pedestrian_Crossing-Human_Control',
       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
       'Year', 'Month', 'Hour', 'Day','Carriageway_Hazards', 
       'Junction_Control', 'Special_Conditions_at_Site','LSOA_of_Accident_Location',
       'Local_Authority_(District)', 'Local_Authority_(Highway)']
labels = ['Fatal']
#%%
data = (data-data.min())/(data.max()-data.min())
#%%
data = data.fillna(0)
#%%
data.sort_values(by=['Year','Month','Day','Hour'],axis=0,inplace=True)

#%%
featuresd = pd.read_csv(r'.\learneddata.csv')
time_format = '%Y/%m/%d'
featuresd['Unnamed: 0'] = pd.to_datetime(featuresd['Unnamed: 0'], format=time_format)
featuresd['Unnamed: 0'] = featuresd['Unnamed: 0'].dt.day.astype(str) + '/' + featuresd['Unnamed: 0'].dt.month.astype(str) + '/' + featuresd['Unnamed: 0'].dt.year.astype(str)

#%%
slight_data = data.groupby(['Date']).agg({'Number_of_Casualties':['sum']})
slight_data.columns = slight_data.columns.get_level_values(0)
#%%
sldata = featuresd.groupby(['Unnamed: 0']).agg({'0':['mean'],'1':['mean'], '2':['mean']})
sldata.columns = sldata.columns.get_level_values(0)
#%%
#%%
fscaler = MinMaxScaler()
sldata2 = fscaler.fit_transform(sldata)
sldatap = pd.DataFrame(data=sldata2,index=sldata.index)
#%%
comb_data = pd.merge(sldatap, slight_data, left_index=True, right_index=True)
#%%
features,labels = data_convert(comb_data,14,4)




#%%
fscaler = MinMaxScaler()
features = fscaler.fit_transform(features)
#%%
lscaler = MinMaxScaler()
labels = lscaler.fit_transform(labels.reshape(-1,1))
#%%
features = np.reshape(features, [features.shape[0], features.shape[1],1])
labels = np.reshape(labels, [labels.shape[0]])
#%%
num_data = labels.shape[0]
train_split = 0.7
valid_split = 0.1
num_train = int(num_data * train_split)
num_valid = int(num_data * valid_split)
num_test = num_data - num_train - num_valid
indices = np.random.permutation(num_data)
#indices = [x for x in range(0,num_data)]
#%%
train_split = (num_data * 70) // 100
valid_split = ((num_data * 10) // 100) + train_split
train_idx, valid_idx, test_idx = indices[:train_split], indices[train_split:valid_split], indices[valid_split:]

features_train = np.array([features[i] for i in train_idx])
features_valid = np.array([features[i] for i in valid_idx])
features_test = np.array([features[i] for i in test_idx])

labels_train = np.array([labels[i] for i in train_idx])
labels_valid = np.array([labels[i] for i in valid_idx])
labels_test = np.array([labels[i] for i in test_idx])
#%%
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((features_train,labels_train)).batch(batch_size).repeat()
valid_dataset = tf.data.Dataset.from_tensor_slices((features_valid,labels_valid)).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((features_test,labels_test)).batch(batch_size).repeat()
#%%
train_steps = (labels_train.shape[0] // batch_size) + 1
valid_steps = (labels_valid.shape[0] // batch_size) + 1
test_steps = (labels_test.shape[0] // batch_size) + 1
#%%
model = keras.Sequential([
    keras.layers.GRU(units=16, input_shape=[features.shape[1],features.shape[2]], return_sequences=True),
    keras.layers.GRU(units=32, return_sequences = True),
    keras.layers.GRU(units=64, return_sequences = False),
    keras.layers.Dense(units=1, activation='linear')
])
#%%
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
#%%
model.summary()
#%%
history = model.fit(train_dataset,steps_per_epoch=train_steps, epochs=100,validation_data=valid_dataset,validation_steps=valid_steps)
#%%
test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps)

print('Test loss:', test_loss)
#%%
predictions = model.predict(test_dataset, steps=test_steps)
#%%
labels_predicted = lscaler.inverse_transform(predictions)
#%%
plt.plot(labels_test[30:50], color='blue')
plt.plot(predictions[30:50], color='red')
plt.show()

