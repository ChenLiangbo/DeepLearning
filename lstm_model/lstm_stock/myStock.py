#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from tensorflow.python.ops import rnn 



class MyLSTM(object):
    def __init__(self,):
        super(MyLSTM,self).__init__()
        self.dir            = os.path.join(os.path.dirname(__file__),'model3/')
        self.path           = self.dir + 'stockModel.ckpt'
        self.time_step      = 20      #时间步
        self.rnn_unit       = 15      #hidden layer units
        self.batch_size     = 60      #每一批次训练多少个样例
        self.input_size     = 7       #输入层维度
        self.output_size    = 1       #输出层维度
        self.lr             = 0.0006  #学习率
        self.epoch          = 10000   #训练周期
        self.X              = tf.placeholder(tf.float32, shape=[None,self.time_step,self.input_size])
        self.Y              = tf.placeholder(tf.float32, shape=[None,self.time_step,self.output_size])

        self.weights        = {
                               'in':tf.Variable(tf.random_normal([self.input_size,self.rnn_unit])),
                               'out':tf.Variable(tf.random_normal([self.rnn_unit,self.output_size]))
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

    def train(self,data):
        batch_index,train_x,train_y = self.get_train_data(data)
        print("train_x = ",len(train_x),len(train_x[0]),"train_y = ",len(train_y),len(train_y[0]))
        pred,_= self.lstm_model(init_state_size = self.batch_size)
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
                for step in range(len(batch_index)-1):
                    x_batch = train_x[batch_index[step]:batch_index[step+1]]
                    x_batch = np.array(x_batch)
                    y_batch = train_y[batch_index[step]:batch_index[step+1]]
                    y_batch = np.array(y_batch)
                    print("x_batch = ",x_batch.shape,"y_batch = ",y_batch.shape)
                    sess.run(train_op,feed_dict = {self.X: x_batch,self.Y: y_batch})
                    _,loss_=sess.run([train_op,loss],feed_dict={self.X:x_batch,self.Y: y_batch})
                    break
                break
                print("i = ",i,"loss_= ",loss_)
            sess.run(pred,feed_dict = {self.X: x_batch})
            saver.save(sess,self.path)

    def prediction(self,data):
        print("="*80)
        mean,std,test_x,test_y = self.get_test_data(data)
        print("test data = ",np.array(test_x[0]).shape,np.array(test_y).shape)
        predict_op,_ = self.lstm_model(init_state_size = 1)     
        with tf.Session() as sess:
            #参数恢复
            sess.run(tf.initialize_all_variables())
            saver=tf.train.Saver()
            saver.restore(sess,self.path)
            # saver.restore(sess,'./model2/model2.ckpt')
            test_predict=[]
            for step in range(len(test_x)-1):
              x_test = np.array([test_x[step]])
              # print("x_test.shape = ",x_test.shape)
              prob = sess.run(predict_op,feed_dict={self.X:x_test})   
              predict = prob.reshape((-1))
              print("predict.shape = ",predict.shape)
              test_predict.extend(predict)
            test_y=np.array(test_y)*std[7]+mean[7]
            test_predict=np.array(test_predict)*std[7]+mean[7]
            acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
            #以折线图表示结果
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b',label = 'test_predict')
            plt.plot(list(range(len(test_y))), test_y,  color='r',label = 'test_y')
            plt.legend()
            plt.grid(True)
            plt.show()

    def read_data(self,filename):
        f  = open(filename) 
        df = pd.read_csv(f)     #读入股票数据
        f.close()
        # print("head = ",df.head())
        data = df.iloc[:,2:10].values  #取第3-10列
        return data

    def get_train_data(self,data,train_begin=0,train_end=5800):
        batch_index = []
        time_step  = self.time_step
        data_train = data[train_begin:train_end]
        normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
        train_x,train_y=[],[]   #训练集 
        for i in range(len(normalized_train_data)-time_step):
           if i % self.batch_size==0:
               batch_index.append(i)
           x=normalized_train_data[i:i+time_step,:7]
           y=normalized_train_data[i:i+time_step,7,np.newaxis]
           train_x.append(x.tolist())
           train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data)-time_step))
        return batch_index,train_x,train_y

    def get_test_data(self,data,test_begin=5800):
        time_step = self.time_step
        data_test=data[test_begin:]
        mean=np.mean(data_test,axis=0)
        std=np.std(data_test,axis=0)
        normalized_test_data=(data_test-mean)/std  #标准化
        # print("normalized_test_data = ",normalized_test_data.shape)
        size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
        # print("size = ",size)
        test_x,test_y=[],[]  
        for i in range(size-1):
           x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
           y=normalized_test_data[i*time_step:(i+1)*time_step,7]
           print("i = ",i,"x = ",x.shape,"y = ",y.shape)
           test_x.append(x.tolist())
           test_y.extend(y)
        print("len(test_x = ",len(test_x),"len(test_y)= " ,len(test_y))
        test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
        test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
        print("len(test_x = ",len(test_x),"len(test_y)= " ,len(test_y))
        return mean,std,test_x,test_y

if __name__ == '__main__':
    filename = './dataset/dataset_2.csv'
    myModel = MyLSTM()
    data = myModel.read_data(filename)
    print("data = ",data.shape)
    batch_index,train_x,train_y = myModel.get_train_data(data)
    mean,std,test_x,test_y = myModel.get_test_data(data)
    # myModel.train(data)
    myModel.prediction(data)