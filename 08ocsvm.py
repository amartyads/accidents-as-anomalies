# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:54:57 2019

@author: Amartya
"""

from sklearn import svm
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
datah1 = data[data.Fatal == 1].sample(700, random_state = 314)
datah2 = data[data.Fatal == 0].sample(n=40000, random_state = 314)
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
clf = svm.OneClassSVM(nu=0.05, kernel="rbf")
clf.fit(train_x)
#%%
y_pred_train = clf.predict(train_x)
y_pred_test = clf.predict(test_x)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
#%%
y_pred_test = [(-x + 1)//2 for x in y_pred_test]
#%%
test_y = test_y.values
#%%
tp = 0
tn = 0
fn = 0
fp = 0
for i in range(0,len(test_y)):
    if(test_y[i] == y_pred_test[i]):
        if test_y[i] == 1:
            tp = tp + 1
        else:
            tn = tn + 1
    else:
        if y_pred_test[i] == 1:
            fp = fp + 1
        else:
            fn = fn + 1
#%%
y_pred_scores = clf.decision_function(test_x)
#%%
error_df = pd.DataFrame({'Predictions': y_pred_scores,
                        'True_class': test_y})
error_df.describe()
#%%
plt.clf()
threshold_fixed = 0.0
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Predictions, marker='o', ms=3.5, linestyle='',
            label= "Fatal" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Outputs of decision function")
plt.ylabel("Result")
plt.xlabel("Data point index")
plt.show();