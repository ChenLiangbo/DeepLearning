#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,Model
from keras.optimizers import Adam

classes = 512   #输出类别
pfull   = 0.5   #全链接层弃权比重
print "strat build model..."
#卷积神经网络结构，使用泛型模型，多输出
inputs = Input(shape= (1,32,32))   #使用生成的子数据集，512类，32x32
layer1 = inputs                                                  #layer1 input
layer2 = Convolution2D(nb_filter=96,nb_row=5,nb_col=5,dim_ordering='th',
	border_mode ='same',activation='relu',bias=True)(layer1)    #layer2 96Conv5
layer3 = MaxPooling2D(pool_size=(3,3), strides=(1,1),dim_ordering='th',            
	border_mode = 'same')(layer2)                                #laery3 96MaxP3
layer4 = Convolution2D(nb_filter=128,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer3)    #layer4 128Conv3
layer5 = Convolution2D(nb_filter=196,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer4)    #layer5 196Conv3
layer6 = Convolution2D(nb_filter=256,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer4)    #layer6 256Conv3
layer7 = MaxPooling2D(pool_size=(3,3), strides=(1,1),           
	border_mode = 'same')(layer6)                                #layer7 256MaxP3
layer8 = Convolution2D(nb_filter=352,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer7)    #layer8 352Conv3
layer9 = Convolution2D(nb_filter=480,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer8)    #layer9 480Conv3
layer10 = Convolution2D(nb_filter=512,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer4)    #layer10 512Conv3
layer11 = MaxPooling2D(pool_size=(3,3), strides=(1,1),           
	border_mode = 'same')(layer10)                               #layer11 256MaxP3                                                 #此处稍有不
layer12 = Convolution2D(nb_filter=512,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer10)    #layer12 512Conv3
layer13 = Convolution2D(nb_filter=640,nb_row=3,nb_col=3,dim_ordering='th',
	border_mode = 'same',activation='relu',bias=True)(layer12)    #layer13 640Conv3
layer14 = MaxPooling2D(pool_size=(3,3), strides=(1,1),           
	border_mode = 'same')(layer13)                                #layer14 640MaxP3
layer15 =  Dense(4096)(layer14)                                   #layer15 4096full
layer15 = Activation('relu')(layer15)
layer15 = Dropout(pfull)(layer15)
layer15 = Dense(classes)(layer15)                                 #out 512softmax
out     = Activation('softmax')(layer15)
shape =layer14.shape
print "model build successfully!"
model = Model(input=inputs, output=out)

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print "compile model okay!"
x_train = np.load('../dataset/kerasSet/x_train.npy')
y_train = np.load('../dataset/kerasSet/y_train512.npy')
print "train data = ",(x_train.shape,y_train.shape)

model.fit(x_train,y_train,bn_epoch = 1,batch_size = 256,
	verbose = 1,shuffle = True,validation_split = 0.1)

print "save model"

json_string = model.to_json()  
open('CNNmodelArchitecture.json','w').write(json_string)  
model.save_weights('CNNmodelWeights.h5') 

y_train = np.load('../dataset/kerasSet/y_train.npy')
y_predict = model.predict(x_train)
y_predict = np.argmax(y_predict,axis = 1)
accuracy = np.mean(x_train == y_predict)
print "validate accuracy = ",accuracy
'''
print "load model"
from keras.models import model_from_json 
model = model_from_json(open('CNNmodelArchitecture.json').read())  
model.load_weights('CNNmodelWeights.h5')  
'''

x_test = np.load('../dataset/kerasSet/x_test.npy')
y_test = np.load('../dataset/kerasSet/y_test.npy')    #标签编号（42845，）

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis = 1)

accuracy = np.mean(y_test == y_predict)
print "test accuracy = ",accuracy