# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:08:01 2019

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
from tensorflow.python.keras.optimizers import RMSprop
#%%
def data_convert(dataset,timesteps):
    maxLim = dataset.shape[0]
    features = [[None] * (timesteps)] * (maxLim - timesteps)
    labels = [None] * (maxLim - timesteps)
    for i in range(0,maxLim - timesteps):
        features[i] = dataset[i:i+timesteps]
        labels[i] = dataset[(i+timesteps)]
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
time_format = '%d/%m/%Y'
data['Date'] = pd.to_datetime(data['Date'], format=time_format)
#%%
slight_data = data[data.Accident_Severity == 3].groupby(['Date']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Date'})
slight_data.columns = slight_data.columns.get_level_values(0)
#%%
features,labels = data_convert(slight_data.Number_of_Casualties,14)
#%%
fscaler = MinMaxScaler()
features = fscaler.fit_transform(features)
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
#indices = np.random.permutation(num_data)
indices = [x for x in range(0,num_data)]
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
    keras.layers.GRU(units=512, input_shape=[features.shape[1],1], return_sequences=False),
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
plt.plot(labels_test, color='blue')
plt.plot(predictions, color='red')
plt.show()
#%%





fatal_data = data[data.Accident_Severity == 1].groupby(['Date']).agg({'Number_of_Casualties':['sum']}).reset_index().rename(columns={'index':'Date'})
fatal_data.columns = fatal_data.columns.get_level_values(0)
#%%
features,labels = data_convert(fatal_data.Number_of_Casualties,14)
#%%
fscaler = MinMaxScaler()
features = fscaler.fit_transform(features)
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
i#ndices = [x for x in range(0,num_data)]
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
    keras.layers.GRU(units=512, input_shape=[features.shape[1],1], return_sequences=False),
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
plt.plot(labels_test, color='blue')
plt.plot(predictions, color='red')
plt.show()