#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import input_data
import time

t0 = time.time()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_biases(shape = [32],name = None):
    init = tf.zeros(shape)
    return tf.Variable(init, name = name)  


def conv2dReLU(x,w,b,strides = [1,1,1,1],padding = 'SAME'):
    conv = tf.nn.conv2d(x, w, strides = strides, padding = padding)
    return tf.nn.relu(tf.nn.bias_add(conv, b))


def max_pool3x3(conv,k = 3):
    return tf.nn.max_pool(conv, ksize=[1, k, k, 1],
                       strides=[1, k, k, 1], padding='SAME')


# 定义卷积神经网络模型
def conv_net(x, p_keep_conv, p_keep_hidden):
    # print "------------------------net---------------------"
    w = init_weights([5, 5, 1, 32])
    b = init_biases([32])
    conv1 = conv2dReLU(x,w,b)
    pool1 = max_pool3x3(conv1,k = 3)
    # pool1 = tf.nn.dropout(pool1, p_keep_conv)    

    w2    = init_weights([3, 3, 32, 64])
    b2    =  init_biases([64])
    conv2 = conv2dReLU(pool1,w2,b2)
    pool2 = max_pool3x3(conv2,k = 3)
    # pool2 = tf.nn.dropout(pool2, p_keep_conv)

    w3    = init_weights([3, 3, 64, 128])
    b3    = init_biases([128])
    conv3 = conv2dReLU(pool2,w3,b3)
    pool3 = max_pool3x3(conv3,k = 3)
    # pool3 = tf.nn.dropout(pool3, p_keep_conv)
    print "pool3 shape = ",pool3.get_shape()

    w4 = init_weights([128 * 2 * 2, 1200])
    b4 = init_biases([1200])
    l3 = tf.reshape(pool3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    full = tf.nn.relu(tf.nn.bias_add(tf.matmul(l3, w4), b4))
    full = tf.nn.dropout(full, p_keep_hidden)

    w_out = init_weights([1200, 10])
    b_out = init_biases([10])
    a_out = tf.nn.bias_add(tf.matmul(full, w_out), b_out)
    out = tf.nn.softmax(a_out)
    print "out shape = ",out.get_shape()
    return out

print "data loading" +"-"*40
# 加载数据
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print "x_train = ",(x_train.shape,y_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print "x_train = ",(x_train.shape,y_train.shape)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
hypothesis = conv_net(X, p_keep_conv, p_keep_hidden)
print "network okay" + "-"*40
learningRate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
train_op = tf.train.RMSPropOptimizer(learningRate, 0.9).minimize(cost)
predict_op = tf.argmax(hypothesis, 1)
correctPrediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "training start ..."+"-"*40
batchSize = 128
iterTimes = 10
shape = x_train.shape
for i in range(iterTimes):
    # print zip(range(0, shape[0], batchSize), range(batchSize,shape[0], batchSize))
    break
    j = 0
    for start, end in zip(range(0, shape[0], batchSize), range(batchSize,shape[0], batchSize)):
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]
        # print "batch shape = ",(x_batch.shape,y_batch.shape)
        sess.run(train_op, feed_dict={X: x_batch, Y: y_batch,
                                      p_keep_conv: 0.4, p_keep_hidden: 0.2})
        # print "j = ",j
        # j = j + 1

        if j % 10 == 0:
            loss, accuracyRate = sess.run([cost, accuracy], feed_dict={X: x_batch,Y: y_batch,
                                                        p_keep_conv: 1.,p_keep_hidden:1.0})
            print "iter = %d,step = %d, loss = %f, accuracy = %f " % (i,j,loss,accuracyRate)
 
            # print j, np.mean(np.argmax(y_batch, axis=1) ==
            #          sess.run(predict_op, feed_dict={X: x_batch, Y: y_batch,
            #                   p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        
                                                    
savePath = './paperModel/paperModel.ckpt'
saver = tf.train.Saver()
saver.save(sess,savePath)


test_indices = np.arange(x_test.shape[0]) # Get A Test Batch
np.random.shuffle(test_indices)
# test_indices = test_indices[0:256]

accuracy = np.mean(np.argmax(y_test[test_indices], axis=1) ==
                     sess.run(predict_op, feed_dict={X: x_test[test_indices],
                                                     p_keep_conv: 1.0,
                                                     p_keep_hidden: 1.0}))

sess.close()
print "accuracy = ",accuracy
from mychat import mychatObjeect
import json
access_token = mychat.getTokenIntime()
message = "CNN model Result About MNIST from paper_tensorflow,accuracy = %f,It takes %f seconds!" %(round(accuracy,5),time.time()-t0)
txt_dict = mychat.sendTxtMsg(access_token,message,'go2newera0006')
txt_dict =　json.loads(txt_dict)     #{u'errcode': 0, u'errmsg': u'ok'}
print "txt_dict = ",txt_dict 