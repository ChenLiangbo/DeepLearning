#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os 
import pickle 


class MyCNNmodel(object):
    
    def __init__(self,):
        super(MyCNNmodel,self).__init__()
        self.drop_out      = 0.5
        self.inputSize     = 28
        self.outputSize    = 10
        self.batchSize     = 128
        self.learningrate  = 0.001
        self.iterTimes     = 2
        self.modeldir      = os.path.join(os.path.dirname(__file__),'CNNmodel/')
        self.filedir       = os.path.join(os.path.dirname(__file__),'file/')
        self.savePath      = self.modeldir + 'cnnModelSix.ckpt'        #模型稳健
        self.parameterPath = self.modeldir + 'cnnParameterSix.txt'     #参数文件

        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)

    
    def init_weights(self,shape ):
        init = tf.random_normal(shape,stddev = 0.1)
        return tf.Variable(init,)


    def init_biases(self,shape):
        init = tf.zeros(shape)
        return tf.Variable(init,)        


    def conv2d(self,x, W, b, strides = 1):
        x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def max_pool3x3(x, k = 2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 1, 1, 1],
                          padding='SAME')


    def conv_net(self,X, p_keep_conv, p_keep_hidden):
        w = self.init_weights([5, 5, 1, 32])
        b = self.init_biases([32])
        l1a = tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME')
        l1a = tf.nn.relu(tf.nn.bias_add(l1a, b))
        l1 = tf.nn.max_pool(l1a, ksize = [1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        l1 = tf.nn.dropout(l1, p_keep_conv)
        
        w2    = self.init_weights([3, 3, 32, 64])
        conv2 = tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME')
        b2    =  self.init_biases([64])
        l2a = tf.nn.relu(tf.nn.bias_add(conv2, b2))
        l2 = tf.nn.max_pool(l2a, ksize  = [1, 2, 2, 1],
                                 strides= [1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, p_keep_conv)

        w3 = self.init_weights([3, 3, 64, 128])
        b3 = self.init_biases([128])
        conv3 = tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME')
        l3a = tf.nn.relu(tf.nn.bias_add(conv3, b3))
        l3 = tf.nn.max_pool(l3a, ksize   = [1, 2, 2, 1],
                                 strides = [1, 2, 2, 1], padding='SAME')
    
        w4 = self.init_weights([128 * 4 * 4, 1200])
        b4 = self.init_biases([1200])
        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
        l3 = tf.nn.dropout(l3, p_keep_conv)
        full = tf.matmul(l3, w4)
    
        l4 = tf.nn.relu(tf.nn.bias_add(full, b4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)
        w_o = self.init_weights([1200, 10])
    
        out = tf.matmul(l4, w_o)
        return out
      
    def train(self,x_train,y_train):
        X = tf.placeholder("float", [None, self.inputSize, self.inputSize, 1])
        Y = tf.placeholder("float", [None, self.outputSize])
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
        
        shape = x_train.shape
        for i in range(self.iterTimes):
            j = 0
            for start, end in zip(range(0, shape[0], self.batchSize), range(self.batchSize , shape[0], self.batchSize )):
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                sess.run(train_op, feed_dict={X:x_batch, Y: y_batch,
                                p_keep_conv: 0.8, p_keep_hidden: 0.5})
                j = j + 1
                if j % 10 == 0:
                    loss, accuracyRate = sess.run([cost, accuracy], feed_dict={X: x_batch,Y: y_batch,
                                                        p_keep_conv: 1.,p_keep_hidden:1.0})
                    print "i = %d,j = %d, loss = %f, accuracy = %f " % (i,j,loss,accuracyRate)

                    # break
            # break
        saver = tf.train.Saver()
        saver.save(sess,self.savePath)
        sess.close()


    def predict(self,x_test,y_test = None):
        X = tf.placeholder("float", [None, self.inputSize, self.inputSize, 1])
        Y = tf.placeholder("float", [None, self.outputSize])
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

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,self.savePath)
        hypo = sess.run(hypothesis,feed_dict = {X:x_test,p_keep_conv: 0.8, p_keep_hidden: 0.5})
        print "-"*80
        print "hypo = ",hypo.shape
        
        np.save(self.filedir + 'hypothesis6',hypo)
        print "-"*80
        if type(y_test) is None:
            y_predict = sess.run(predict_op,feed_dict={X: x_test,
                              p_keep_conv: 1.0, p_keep_hidden: 1.0})
        else:
            y_predict = sess.run(accuracy, feed_dict={X: x_test,Y: y_test,
                               p_keep_conv: 1.0, p_keep_hidden: 1.0})
            sess.close()

        return y_predict

if __name__ == '__main__':

    import numpy as np
    import input_data

    trainNumber = 5000
    testNumber  = 2000
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    print "x_train.shape = ",x_train.shape
    x_train = x_train[0:trainNumber]
    y_train = y_train[0:trainNumber]
    x_test = x_test[0:testNumber]
    y_test = y_test[0:testNumber]
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    
    print "x_train.shape = ",x_train.shape    
    myModel = MyCNNmodel()
    myModel.train(x_train,y_train)
    
