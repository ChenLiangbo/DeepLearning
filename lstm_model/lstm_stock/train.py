#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


f  = open('./dataset/5years_day.csv') 
df = pd.read_csv(f)     #读入股票数据
print("head = ",df.head())
data = df.iloc[:,1:6].values  #取第3-10列
shape = data.shape
print("shape = ",data.shape)
Open = list(df['Open'])
Open.reverse()

High = list(df['High'])
High.reverse()
Close = list(df['Close'])
Close.reverse()
Low = list(df['Low'])
Low.reverse()
Volume = list(df['Volume'])
Volume.reverse()
Open = np.array(Open).reshape((shape[0],1))
High = np.array(High).reshape((shape[0],1))
Close = np.array(Close).reshape((shape[0],1))
Low   = np.array(Low).reshape((shape[0],1))
Volume = np.array(Volume).reshape((shape[0],1))
print("Open.shape = ",Open.shape)

x = np.hstack([Open,High,Low,Volume,Close])
y = np.array(Close)
print("x.shape = ",x.shape,"y.shape = ",y.shape)

np.save("./model3/x",x)
np.save('./model3/y',y)

x = np.load("./model3/x.npy")
y = np.load("./model3/y.npy")
shape = (1258, 5)
time_step = 20
x_sample = []
y_sample = []
for i in range(shape[0] - time_step*2):
    x_array = x[i:i+time_step,:]
    y_array = y[i+time_step:i+time_step*2,:]
    # print("array shape = ",x_array.shape,y_array.shape)
    x_sample.append(x_array)
    y_sample.append(y_array)
    # print("i = ",i)

    # break

x_sample = np.asarray(x_sample)
y_sample = np.asarray(y_sample)

'''
xmax = np.amax(x_sample, axis=0)
xmin = np.amin(x_sample, axis=0)
x_sample = (x_sample - xmin) / (xmax - xmin)

xmax = np.amax(y_sample, axis=0)
xmin = np.amin(y_sample, axis=0)
y_sample = (y_sample - xmin) / (xmax - xmin)
'''
x_sample=(x_sample-np.mean(x_sample,axis=0))/np.std(x_sample,axis=0)
y_sample=10*(y_sample-np.mean(y_sample,axis=0))/np.std(y_sample,axis=0)


print("x_sample.shape = ",x_sample.shape,y_sample.shape)
np.save("./model3/x_sample",x_sample)
np.save('./model3/y_sample',y_sample)

train_end = 1000
x_train = x_sample[0:train_end,:,:]
y_train = y_sample[0:train_end,:,:]
x_test  = x_sample[train_end:,:,:]
y_test  = y_sample[train_end:,:,:]
print("x_train = ",x_train.shape,y_train.shape)

from model import MyLSTM

model = MyLSTM(layers = [5,10,1],time_step = time_step,lr = 0.0006,epoch = 5)

print("train LSTM model ...")
# model.train(x_train,y_train)

print("evaluate LSTM model ...")
model.evaluate(x_test,y_test)