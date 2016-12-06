#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import input_data
from cnnModel import MyCNNmodel

testNumber  = 10000
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x_test = x_test[0:testNumber]
y_test = y_test[0:testNumber]
x_test  = x_test.reshape(-1, 28, 28, 1)

print "x_test.shape = ",(x_test.shape,y_test.shape)

print "y_test = ",y_test[0]
myModel = MyCNNmodel()

accuracy = myModel.predict(x_test,y_test)
print "accuracy = ",accuracy
from mychat import mychatObject

access_token = mychatObject.getTokenIntime()

content = {"accuracy":accuracy,"testNumber":testNumber}

message = "mnist CNN model predict result demo-p2,testNumber = %d,accracy = %f" % (testNumber,accuracy)
txt_dict = mychatObject.sendTxtMsg(access_token,message,'go2newera0006')
txt_dict = simplejson.loads(txt_dict)     #{u'errcode': 0, u'errmsg': u'ok'}
print "txt_dict = ",txt_dict 
import json

fp = open('./file/mnist_result.txt','ab')
content = json.dumps(content)
fp.write(content)
fp.write('\r\n')
fp.close()    
print "It is ok!"
