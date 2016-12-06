#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np


input_shape = (1,32, 32)
pout = 0.5
model = Sequential()
layer2 = Convolution2D(nb_filter=96,nb_row=5,nb_col=5,dim_ordering='th',input_shape = input_shape,
	border_mode ='same',activation='relu',bias=True)    #layer2 96Conv5
model.add(layer2)
layer3 = MaxPooling2D(pool_size=(3,3), strides=(1,1),dim_ordering='th',            
	border_mode = 'same')                             #laery3 96MaxP3
model.add(layer3)
layer4 = Convolution2D(nb_filter=128,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer4 128Conv3
model.add(layer4)
layer5 = Convolution2D(nb_filter=196,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer5 196Conv3
model.add(layer5)
layer6 = Convolution2D(nb_filter=256,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer6 256Conv3
model.add(layer6)
layer7 = MaxPooling2D(pool_size=(3,3), strides=(1,1),           
	border_mode = 'same')                               #layer7 256MaxP3
model.add(layer7)
layer8 = Convolution2D(nb_filter=352,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer8 352Conv3
model.add(layer8)
layer9 = Convolution2D(nb_filter=480,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer9 480Conv3
model.add(layer9)
layer10 = Convolution2D(nb_filter=512,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer10 512Conv3
model.add(layer10)
layer11 = MaxPooling2D(pool_size=(3,3), strides=(1,1),           
	border_mode = 'same')                               #layer11 256MaxP3
	                                                             #此处稍有不
model.add(layer11)
layer12 = Convolution2D(nb_filter=512,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer12 512Conv3
model.add(layer12)
layer13 = Convolution2D(nb_filter=640,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)    #layer13 640Conv3
model.add(layer13)
layer14 = MaxPooling2D(pool_size=(3,3), strides=(1,1),
	border_mode = 'same')          
model.add(layer14)

model.add(Dropout(pout))
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(pout))
model.add(Dense(512))
model.add(Activation('softmax'))

print "compile model"
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
nb_classes = 512
x_train = np.load('../dataset/kerasSet/x_train.npy')
y_train = np.load('../dataset/kerasSet/y_train.npy')
x_train = x_train.reshape(-1,1,32,32)
y_train = np_utils.to_categorical(y_train, nb_classes)

print "train data = ",(x_train.shape,y_train.shape)

model.fit(x_train,y_train,nb_epoch = 1,batch_size = 256,
	verbose = 1,shuffle = True,validation_split = 0.1)

print "save model"

json_string = model.to_json()  
open('CNNmodelArchitecture2.json','w').write(json_string)  
model.save_weights('CNNmodelWeights2.h5') 

y_train = np.load('../dataset/kerasSet/y_train.npy')
y_predict = model.predict(x_train)
y_predict = np.argmax(y_predict,axis = 1)
accuracy = np.mean(x_train == y_predict)
print "validate accuracy = ",accuracy