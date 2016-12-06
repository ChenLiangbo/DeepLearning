#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from CNNPaperModel import MyCNNmodel
import input_data
import time

t0 = time.time()
trainNumber = 100
testNumber  = 10000
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print "x_train.shape = ",x_train.shape
x_train = x_train[0:trainNumber]
y_train = y_train[0:trainNumber]
x_test = x_test[0:testNumber]
y_test = y_test[0:testNumber]
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

print "x_train.shape = ",x_train.shape

print "y_test = ",y_test[0]
myModel = MyCNNmodel()

accuracy = myModel.predict(x_test,y_test)
print "accuracy = ",accuracy
from mychat import mychatObjeect
import json
access_token = mychatObjeect.getTokenIntime()
message = "CNN model Result About MNIST from mnist/p2-demo/predict on 41 server,\
          accuracy = %f,It takes %f seconds!" %(round(accuracy,5),time.time()-t0)
txt_dict = mychatObjeect.sendTxtMsg(access_token,message,'go2newera0006')
txt_dict = json.loads(txt_dict)     #{u'errcode': 0, u'errmsg': u'ok'}
print "txt_dict = ",txt_dict 
