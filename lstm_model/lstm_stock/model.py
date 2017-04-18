#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from tensorflow.python.ops import rnn 
from matplotlib import pyplot as plt


class MyLSTM(object):
    def __init__(self,layers = [4,15,1],time_step = 20,lr = 0.0006,epoch = 5 ):
        super(MyLSTM,self).__init__()
        self.dir            = os.path.join(os.path.dirname(__file__),'model3/')
        self.path           = self.dir + 'stockModel.ckpt'
        self.time_step      = 20      #时间步
        self.rnn_unit       = layers[1]      #hidden layer units
        self.batch_size     = 60      #每一批次训练多少个样例
        self.input_size     = layers[0]       #输入层维度
        self.output_size    = layers[2]       #输出层维度
        self.lr             = 0.0006  #学习率
        self.epoch          = epoch   #训练周期
        self.X              = tf.placeholder(tf.float32, shape=[None,self.time_step,self.input_size])
        self.Y              = tf.placeholder(tf.float32, shape=[None,self.time_step,self.output_size])

        self.weights        = {
                               'in':tf.Variable(tf.truncated_normal([self.input_size,self.rnn_unit], stddev = 0.1)),
                               'out':tf.Variable(tf.truncated_normal([self.rnn_unit,self.output_size],stddev = 0.1))
                               }

        self.biases         = {
                               'in':tf.Variable(tf.constant(0.1,shape=[self.rnn_unit,])),
                               'out':tf.Variable(tf.constant(0.1,shape=[self.output_size,]))
                              }


    def read_data(self,filename):
        pass

    def lstm_model(self,init_state_size):
        w_in = self.weights['in']
        b_in = self.biases['in'] 
        # print("lstm X = ",X.get_shape()) 
        my_input = tf.reshape(self.X,[-1,self.input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        # print("my_input.shape = ",my_input.get_shape())
        input_rnn = tf.nn.tanh(tf.matmul(my_input,w_in)+b_in)
        # print("input_rnn.shape ----1 = ",input_rnn.get_shape())
        input_rnn = tf.reshape(input_rnn,[-1,self.time_step,self.rnn_unit])  #将tensor转成3维，作为lstm cell的输入
        # print("input_rnn.shape ----2 = ",input_rnn.get_shape())
        cell=tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
        init_state = cell.zero_state(init_state_size,dtype=tf.float32)
        output_rnn,final_states = rnn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output = tf.reshape(output_rnn,[-1,self.rnn_unit]) #作为输出层的输入
        # print("output.shape = ",output.get_shape())
        w_out = self.weights['out']
        b_out = self.biases['out']
        predict = tf.matmul(output,w_out)+b_out
        # print("predict.shape = ",predict.get_shape())
        # print("-"*30 + ' lstm ' + '-'*30)
        return predict,final_states

    def train(self,x_train,y_train):
        shape = x_train.shape   # (1000,20,4)  (1000,20,1)
        
        pred,_= self.lstm_model(init_state_size = 1)
        #损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(self.Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            #重复训练10000次
            try:
                saver.restore(sess,self.path)
            except Exception as ex:
                print("[Exception Information] ",str(ex))
            for i in range(self.epoch):
                for step in range(shape[0]):
                    x_batch = x_train[step:step+1,:,:]
                    y_batch = y_train[step:step+1,:,:]
                    # print("x_batch = ",x_batch.shape,"y_batch = ",y_batch.shape)
                    sess.run(train_op,feed_dict = {self.X: x_batch,self.Y: y_batch})
                    _,loss_=sess.run([train_op,loss],feed_dict={self.X:x_batch,self.Y: y_batch})
                    # break
                # break
                print("i = ",i,"loss_= ",loss_)
            sess.run(pred,feed_dict = {self.X: x_batch})
            saver.save(sess,self.path)


    def evaluate(self,x_test,y_test):
        shape = x_test.shape  # (N,20,4)  (N,20,1)
        predict_op,_ = self.lstm_model(init_state_size = 1)     
        with tf.Session() as sess:
            #参数恢复
            sess.run(tf.initialize_all_variables())
            saver=tf.train.Saver()
            saver.restore(sess,self.path)
            # saver.restore(sess,'./model2/model2.ckpt')
            y_real = []
            y_pred = []
            for step in range(shape[0]):
              # print("step = ",step)
              x = x_test[step:step+1,:,:]
              y = y_test[step:step+1,:,:].reshape((-1))
              # print("x.shape = ",x.shape,y.shape)
              prob = sess.run(predict_op,feed_dict={self.X:x})   
              predict = prob.reshape((-1))
              # print("predict.shape = ",predict.shape)
              y_real.append(y[1])
              y_pred.append(predict[1])
            
            #以折线图表示结果
            
            plt.figure()
            plt.plot(list(range(len(y_real))),y_real,color='r',label = 'y_real')
            plt.plot(list(range(len(y_pred))),y_pred,color='b',label = 'y_pred')
            plt.legend()
            plt.grid(True)
            plt.show()

    def predict(self,x_test):
        predict_op,_ = self.lstm_model(init_state_size = 1)     
        with tf.Session() as sess:
            #参数恢复
            sess.run(tf.initialize_all_variables())
            saver=tf.train.Saver()
            saver.restore(sess,self.path)
            # saver.restore(sess,'./model2/model2.ckpt')
            test_predict=[]
            y_predict = []
            for step in range(len(test_x)-1):
                x = x_test[step:step+1,:,:]
                y = y_test[step:step+1,:,:].reshape((-1))
                # print("x.shape = ",x.shape,y.shape)
                prob = sess.run(predict_op,feed_dict={self.X:x}) 
                y_predict.append(prob)
        return np.array(y_predict)


if __name__ == '__main__':
    filename = './dataset/dataset_2.csv'
    myModel = MyLSTM()
    data = myModel.read_data(filename)
    print("data = ",data.shape)
    batch_index,train_x,train_y = myModel.get_train_data(data)
    mean,std,test_x,test_y = myModel.get_test_data(data)
    # myModel.train(data)
    myModel.prediction(data)