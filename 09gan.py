# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:06:31 2019

@author: Amartya
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from IPython import display
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
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
RANDOM_SEED = 314
TEST_PCT = 0.2
#%%
datah1 = data[data.Fatal == 1]#.sample(500, random_state = 314)
datah2 = data[data.Fatal == 0].sample(n=800000, random_state = RANDOM_SEED)
datas = pd.concat([datah1,datah2])
#%%
train_x, test_x = train_test_split(datas, test_size=TEST_PCT, random_state=RANDOM_SEED)
train_x = train_x[train_x.Fatal == 0] #where normal transactions
train_x = train_x.drop(['Fatal'], axis=1) #drop the class column


test_y = test_x['Fatal'] #save the class column for the test set
test_x = test_x.drop(['Fatal'], axis=1) #drop the class column

train_x = train_x.values #transform to ndarray
test_x = test_x.values
#%%
#convert to images
trainrows = train_x.shape[0]
train_x = np.c_[train_x, np.zeros(trainrows), np.zeros(trainrows) ,np.zeros(trainrows), np.zeros(trainrows)]
train_x = np.reshape(train_x, (trainrows, 5, 5, 1))

testrows = test_x.shape[0]
test_x = np.c_[test_x, np.zeros(testrows), np.zeros(testrows) ,np.zeros(testrows), np.zeros(testrows)]
test_x = np.reshape(test_x, (testrows, 5, 5, 1))

#%%
opt = keras.optimizers.Adam(lr=1e-4)
dopt = keras.optimizers.Adam(lr=1e-3)
#%%
#Generative model
dropout_rate = 0.4
#input: 50 output: 5 x 5 x 1
generator = keras.Sequential([
        keras.layers.Dense(input_dim=100, units=2*2*128),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Reshape((2,2,128)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(64, kernel_size=2, padding='same'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(32, kernel_size=2, padding='same'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Conv2DTranspose(1, kernel_size=2, padding='same'),
        keras.layers.Cropping2D(((2,1),(2,1))),
        keras.layers.Activation('sigmoid')
        ])
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()
#%%
#Discriminative model
dropout_rate = 0.25
#input: 5 x 5 x 1 output: 2 x 2 x 256 -> 1 sigmoid
discriminator = keras.Sequential([
        keras.layers.ZeroPadding2D(input_shape=[5,5,1],padding=((2,1),(2,1))),
        keras.layers.Conv2D(filters=64, kernel_size=2, strides=2),
        keras.layers.LeakyReLU(0.4),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv2D(filters=128, kernel_size=2, strides=2),
        keras.layers.LeakyReLU(0.4),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv2D(filters=256, kernel_size=2, strides=2),
        keras.layers.LeakyReLU(0.4),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.Dense(2, activation='softmax')
        ])
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

#%%
#Freeze weights in discriminator
discriminator.trainable = False
#%%
#Stack model
gan_input = keras.layers.Input(shape=[100])
gan_gen = generator(gan_input)
gan_dis = discriminator(gan_gen)
ganmodel = keras.Model(gan_input, gan_dis)
ganmodel.compile(loss='categorical_crossentropy', optimizer=opt)
#%%
ganmodel.summary()
#%%
ntrain = 10000
trainidx = random.sample(range(0,train_x.shape[0]), ntrain)
XT = train_x[trainidx,:,:,:]
#%%
# Pre-train the discriminator network
noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
generated_images = generator.predict(noise_gen)
#%%
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1
#%%
discriminator.trainable = True
discriminator.fit(X,y, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X)
#%%
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
#diff = y - y_hat
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))
#%%
def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10,8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()
#%%
def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
#%%
losses = {"d":[], "g":[]}
#%%
def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = train_x[np.random.randint(0,train_x.shape[0],size=BATCH_SIZE),:,:,:]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        #make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:] = 1
        
        #make_trainable(discriminator,False)
        g_loss = ganmodel.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_loss(losses)
            #plot_gen()
#%%
train_for_n(nb_epoch=8000, plt_frq=1000,BATCH_SIZE=32)
#%%
plot_loss(losses)
#%%





#%%
test_x_predictions = discriminator.predict(test_x)
test_x_probs = test_x_predictions[:,1]
test_x_labels =  (test_x_probs > 0.5).astype(np.int)
#%%
test_y = test_y.values
#%%
conf = confusion_matrix(test_y, test_x_labels)
#%%
tp = 0
tn = 0
fn = 0
fp = 0
for i in range(0,len(test_y)):
    if(test_y[i] == test_x_labels[i]):
        if test_y[i] == 1:
            tp = tp + 1
        else:
            tn = tn + 1
    else:
        if test_x_labels[i] == 1:
            fp = fp + 1
        else:
            fn = fn + 1
#%%
fpr, tpr, _ = roc_curve(test_y, test_x_probs)
roc_auc = auc(fpr, tpr)
#%%
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()