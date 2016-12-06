#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os
import input_data


filedir = './npyfile/'
filelist = os.listdir(filedir)

y_predict = []
print "start ..."
for f in filelist:
    if "hypo" in f:
        print "filename = ",f
        filename = filedir + f
        hypo = np.load(filename)
        y_predict.append(hypo)

print "hypo = ",(len(y_predict),y_predict[0].shape)
shape = y_predict[0].shape
y1 = []
y2 = []
y3 = []
for i in xrange(shape[0]):
    voted = {}
    for j in xrange(5):
        # print "y_predict[j][i,:] = ",y_predict[j][i,:]
        key = max(y_predict[j][i,:])
        key = list(y_predict[j][i,:]).index(key)
        if key not in voted:
            voted[key] = 1
        else:
            voted[key] = voted[key] + 1
    flag = False
    for key in voted:
        # print "voted key = %d,value = %d" % (key,voted[key])
        if voted[key] >= 3:
            flag = True
            ret = key
            break
    if not flag:
        y1add = y_predict[0][i,:] + y_predict[1][i,:] + y_predict[2][i,:] + y_predict[3][i,:] + y_predict[4][i,:]
        key = max(y1add) 
        ret = list(y1add).index(key)

    y1.append(ret)

    y2add = y_predict[0][i,:] + y_predict[1][i,:] + y_predict[2][i,:] + y_predict[3][i,:] + y_predict[4][i,:]
    y2key = max(y2add)
    y2ret = list(y2add).index(y2key)
    y2.append(y2ret)

    y3value = max(voted.values())
    for k in voted:
        if voted[k] == y3value:
            y3ret = k
            break
    y3.append(y3ret)
        

y_vote1 = np.array(y1).reshape(shape[0],1)
print "y_vote1 = ",y_vote1.shape
print y_vote1[0:10]
y_vote2 = np.array(y2).reshape(shape[0],1)
y_vote3 = np.array(y3).reshape(shape[0],1)

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print "y_test = ",y_test.shape
yreal = np.argmax(y_test,1).reshape(y_test.shape[0],1)
print "yreal = ",yreal.shape
print yreal[0:10]

accuracy1 = np.mean(y_vote1 == yreal)
print "vote1 accuracy = ",accuracy1
print "okay"
accuracy2 = np.mean(y_vote2 == yreal)
print "vote2 accuracy = ",accuracy2
print "okay"

accuracy3 = np.mean(y_vote3 == yreal)
print "vote3 accuracy = ",accuracy3
print "okay"