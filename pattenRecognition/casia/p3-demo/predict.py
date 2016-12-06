#!usr/bin/env/python 
# -*- coding: utf-8 -*-
from CNNPaperModel import MyCNNmodel
import time

t0 = time.time()

datadir = './tensordata/'

x_test = np.load(datadir + 'x_test.npy')
y_test = np.load(datadir + 'y_test.npy')
print "x_test.shape = ",(x_test.shape,y_test.shape)

myModel = MyCNNmodel()

accuracy = myModel.predict(x_test,y_test)
print "accuracy = ",accuracy
from mychat import mychatObjeect
import json
access_token = mychatObjeect.getTokenIntime()
message = "CNN model Result About MNIST from p3-demo on 41 server,\
          accuracy = %f,It takes %f seconds!" %(round(accuracy,5),time.time()-t0)
txt_dict = mychatObjeect.sendTxtMsg(access_token,message,'go2newera0006')
txt_dict = json.loads(txt_dict)     #{u'errcode': 0, u'errmsg': u'ok'}
print "txt_dict = ",txt_dict 
