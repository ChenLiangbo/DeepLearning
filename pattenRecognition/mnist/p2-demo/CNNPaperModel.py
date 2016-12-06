#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os 
import pickle 
import json


class MyCNNmodel(object):
    
    def __init__(self,):
        super(MyCNNmodel,self).__init__()
        self.drop_out      = 0.5
        self.inputSize     = 28
        self.outputSize    = 10
        self.batchSize     = 128
        self.learningrate  = 0.01
        self.iterTimes     = 300
        self.modeldir      = os.path.join(os.path.dirname(__file__),'CNNmodel/')
        self.savePath      = self.modeldir + 'ModelFive.ckpt'        #模型稳健
        self.parameterPath = self.modeldir + 'ParameterFive.txt'     #参数文件

        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)

    def init_weights(self,shape ):
        init = tf.random_normal(shape,stddev = 0.1)
        return tf.Variable(init,)


    def init_biases(self,shape):
        init = tf.zeros(shape)
        return tf.Variable(init,)      

    def conv2dReLU(self,x,w,b,strides = [1,1,1,1],padding = 'SAME'):
        conv = tf.nn.conv2d(x, w, strides = strides, padding = padding)
        return tf.nn.relu(tf.nn.bias_add(conv, b))


    def max_pool3x3(self,conv,k = 3):
        return tf.nn.max_pool(conv, ksize=[1, k, k, 1],
                       strides=[1, k, k, 1], padding='SAME')


    def conv_net(self,x, p_keep_conv, p_keep_hidden):
        w = self.init_weights([5, 5, 1, 32])
        b = self.init_biases([32])
        conv1 = self.conv2dReLU(x,w,b)
        pool1 = self.max_pool3x3(conv1,k = 2)
        # pool1 = tf.nn.dropout(pool1, p_keep_conv)    
    
        w2    = self.init_weights([3, 3, 32, 64])
        b2    = self.init_biases([64])
        conv2 = self.conv2dReLU(pool1,w2,b2)
        pool2 = self.max_pool3x3(conv2,k = 2)
        # pool2 = tf.nn.dropout(pool2, p_keep_conv)
    
        w3    = self.init_weights([3, 3, 64, 128])
        b3    = self.init_biases([128])
        conv3 = self.conv2dReLU(pool2,w3,b3)
        pool3 = self.max_pool3x3(conv3,k = 2)
        print "pool3 shape = ",pool3.get_shape()
    
        w4 = self.init_weights([128 * 4 * 4, 1200])
        b4 = self.init_biases([1200])
        l3 = tf.reshape(pool3, [-1, w4.get_shape().as_list()[0]])
        l3 = tf.nn.dropout(l3, p_keep_conv)
    
        full = tf.nn.relu(tf.nn.bias_add(tf.matmul(l3, w4), b4))
        full = tf.nn.dropout(full, p_keep_hidden)

        w_out = self.init_weights([1200, 10])
        b_out = self.init_biases([10])
        a_out = tf.nn.bias_add(tf.matmul(full, w_out), b_out)
        out = tf.nn.softmax(a_out)
        return out


    def train(self,x_train,y_train):
        X = tf.placeholder("float", [None, 28, 28, 1])
        Y = tf.placeholder("float", [None, 10])
        
        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")
        hypothesis = self.conv_net(X, p_keep_conv, p_keep_hidden)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
        train_op = tf.train.RMSPropOptimizer(self.learningrate, 0.9).minimize(cost)
        predict_op = tf.argmax(hypothesis, 1)
        correctPrediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        batchSize = self.batchSize
        xtrain = x_train
        del x_train

        for i in range(self.iterTimes):
            j = 0
            for start, end in zip(range(0, len(xtrain), batchSize), range(batchSize, len(xtrain), batchSize)):
                x_batch = xtrain[start:end]
                y_batch = y_train[start:end]
                # print "batch shape = ",(x_batch.shape,y_batch.shape)
                sess.run(train_op, feed_dict={X: x_batch, Y: y_batch,
                                          p_keep_conv: 0.4, p_keep_hidden: 0.2})

                if j % 10 == 0:
                    loss, accuracyRate = sess.run([cost, accuracy], feed_dict={X: x_batch,Y: y_batch,
                                                            p_keep_conv: 1.,p_keep_hidden:1.0})
                    adict = {'iter':i,'step':j,'loss':round(loss,4),'accuracy':round(accuracyRate,4)}
                    self.save_iteration(adict)
                    print "iter = %d,step = %d, loss = %f, accuracy = %f " % (i,j,loss,accuracyRate)
                j = j + 1
        saver = tf.train.Saver()
        saver.save(sess,self.savePath)
        sess.close()

    def predict(self,x_test,y_test):
        X = tf.placeholder("float", [None, 28, 28, 1])
        Y = tf.placeholder("float", [None, 10])
        
        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")
        hypothesis = self.conv_net(X, p_keep_conv, p_keep_hidden)
        print "predict net okay" +"-"*40
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
        train_op = tf.train.RMSPropOptimizer(self.learningrate, 0.9).minimize(cost)
        predict_op = tf.argmax(hypothesis, 1)
        correctPrediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,self.savePath)
        print "predict load session okay"+"-"*40
        hypo = sess.run(hypothesis,feed_dict = {X: x_test,p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print "hypo.shape = ",hypo.shape
        np.save('./npyfile/hypothesis5',hypo)
        if type(y_test) is None:
            y_predict = sess.run(predict_op,feed_dict={X: x_test,
                              p_keep_conv: 1.0, p_keep_hidden: 1.0})
        else:
            y_predict = sess.run(accuracy, feed_dict={X: x_test,Y: y_test,
                               p_keep_conv: 1.0, p_keep_hidden: 1.0})
        sess.close()

        return y_predict

    def save_iteration(self,adict):
        adict = json.dumps(adict)
        fp = open(self.parameterPath,'ab')
        fp.write(adict)
        fp.close()
