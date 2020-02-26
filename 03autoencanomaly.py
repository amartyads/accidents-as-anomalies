# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:37:23 2019

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
data.drop(['date_time','Date','Time','Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude',
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
data.drop(['Accident_Severity'], axis=1, inplace=True)
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
#%%
#data['Longitude'] = round(data['Longitude'],1)
#data['Latitude'] = round(data['Latitude'],1)
#data['Location_Easting_OSGR'] = round(data['Location_Easting_OSGR'],1)
#data['Location_Northing_OSGR'] = round(data['Location_Northing_OSGR'],1)
#%%
data.sort_values(by=['Year','Month','Day','Hour'],axis=0,inplace=True)
#%%
datah1 = data[data.Fatal == 1]
datah2 = data[data.Fatal == 0].sample(n=800000)
datas = pd.concat([datah1,datah2])
#%%
RANDOM_SEED = 314 #used to help randomly select the data points
TEST_PCT = 0.2
#%%
train_x, test_x = train_test_split(datas, test_size=TEST_PCT, random_state=RANDOM_SEED)
train_x = train_x[train_x.Fatal == 0] #where normal transactions
train_x = train_x.drop(['Fatal'], axis=1) #drop the class column


test_y = test_x['Fatal'] #save the class column for the test set
test_x = test_x.drop(['Fatal'], axis=1) #drop the class column

train_x = train_x.values #transform to ndarray
test_x = test_x.values
#%%
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((train_x,train_x)).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_x,test_x)).batch(batch_size).repeat()
#%%
nb_epoch = 200
input_dim = train_x.shape[1]
encoding_dim = 30
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-7

#%%
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[input_dim]),
    keras.layers.Dense(units=encoding_dim, activity_regularizer=keras.regularizers.l1(learning_rate), activation='tanh'),
    keras.layers.Dense(hidden_dim, activation="tanh"),
    keras.layers.Dense(hidden_dim/2, activation="tanh"),
    keras.layers.Dense(hidden_dim/4, activation="tanh"),
    keras.layers.Dense(hidden_dim/2, activation="tanh"),
    keras.layers.Dense(hidden_dim, activation='tanh'),
    keras.layers.Dense(encoding_dim, activation='tanh'),
    keras.layers.Dense(input_dim, activation='sigmoid')
])
#%%
model.compile('adam','mean_squared_error', metrics=['accuracy'])
#%%
model.summary()
#%%
stepsno = (train_x.shape[0] // batch_size) + 1
history = model.fit(train_dataset,steps_per_epoch=stepsno, epochs=nb_epoch)
#%%
plt.plot(history.history['loss'], linewidth=2, label='Train')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
#%%
test_x_predictions = model.predict(test_x)
mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': test_y})
error_df.describe()
#%%
outliers = error_df[error_df.Reconstruction_error > 0.1]
fatals = error_df[error_df.True_class == 1]
inliers = error_df[error_df.Reconstruction_error < 0.1]
nonfatals = error_df[error_df.True_class == 0]
#%%
plt.clf()
threshold_fixed = 0.1
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fatal" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();
#%%
plt.savefig('test2png.png', dpi=100)
#%%
model.save('./autoencoder.h5')
#%%
model = keras.models.load_model('./autoencoder.h5')
#%%
model2 = keras.Sequential([
    keras.layers.InputLayer(input_shape=[input_dim]),
    keras.layers.Dense(weights=model.layers[0].get_weights(), units=encoding_dim, activity_regularizer=keras.regularizers.l1(learning_rate), activation='tanh'),
    keras.layers.Dense(weights=model.layers[1].get_weights(), units=hidden_dim, activation="tanh"),
    keras.layers.Dense(weights=model.layers[2].get_weights(), units=hidden_dim/2, activation="tanh"),
    keras.layers.Dense(weights=model.layers[3].get_weights(), units=hidden_dim/4, activation="tanh")
])
#%%
model2.summary()
#%%
labs = data['Fatal']
data2s = data.drop('Fatal', axis=1)
#%%
representations = model2.predict(data2s)
#%%
data2s['Year'] = ((data2s['Year'] * (9)) + 2005).astype(int)
#%%
data2s['Month'] = ((data2s['Month'] * (11)) + 1).astype(int)
#%%
data2s['Day'] = ((data2s['Day'] * (30)) + 1).astype(int)
#%%
datadates = data2s['Year'].astype(str) + '/' + data2s['Month'].astype(str) + '/' + data2s['Day'].astype(str)
#%%
learnneddata = pd.DataFrame(data=representations, index=datadates)
#%%
learnneddata = pd.concat([learnneddata,labs], axis=1)
#%%
learnneddata.to_csv(r'.\learneddata.csv')
#%%
