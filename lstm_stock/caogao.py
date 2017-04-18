#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

'''
f  = open('./dataset/5years_day.csv') 
df = pd.read_csv(f)     #读入股票数据
print("head = ",df.head())
data = df.iloc[:,1:6].values  #取第3-10列
shape = data.shape
print("shape = ",data.shape)
Open = np.array(list(df['Open'])).reshape((shape[0],1))
print("Open.shape = ",Open.shape)
High = np.array(list(df['High'])).reshape((shape[0],1))
Close = np.array(list(df['Close'])).reshape((shape[0],1))
Low   = np.array(list(df['Low'])).reshape((shape[0],1))
Volume = np.array(list(df['Volume'])).reshape((shape[0],1))
x = np.hstack([Open,High,Low,Volume])
y = np.array(Close)
print("x.shape = ",x.shape,"y.shape = ",y.shape)

np.save("./model3/x",x)
np.save('./model3/y',y)
'''
x = np.load("./model3/x.npy")
y = np.load("./model3/y.npy")
shape = (1258, 5)
time_step = 20
x_sample = []
y_sample = []
for i in range(shape[0] - time_step - 1):
    x_array = x[i:i+time_step,:]
    y_array = y[i+1:i+time_step+1,:]
    # print("array shape = ",x_array.shape,y_array.shape)
    x_sample.append(x_array)
    y_sample.append(y_sample)

    # break

x_sample = np.array(x_sample)
y_sample = np.array(y_sample)
print("x_sample.shape = ",x_sample.shape,y_sample.shape)
np.save("./model3/x_sample",x_sample)
np.save('./model3/y_sample',y_sample)
