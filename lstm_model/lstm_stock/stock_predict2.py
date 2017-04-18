#!usr/bin/env/python 
# -*- coding: utf-8 -*-
'''
Created on 2017年2月20日

@author: Lu.yipiao
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn 



#定义常量
rnn_unit    = 10             # hidden layer units
input_size  = 7
output_size = 1
lr          = 0.0006         # 学习率
epoch       = 10000
batch_size  = 60
time_step   = 20             # 时间步长

#——————————————————导入数据——————————————————————
f  = open('./dataset/dataset_2.csv') 
df = pd.read_csv(f)     #读入股票数据
# print("head = ",df.head())
data = df.iloc[:,2:10].values  #取第3-10列
print("data = ",data.shape)

#获取训练集
def get_train_data(train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:7]
       y=normalized_train_data[i:i+time_step,7,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(test_begin=5800):
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


#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置
X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
print("X.shape = ",X.get_shape(),'Y.shape = ',Y.get_shape())

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }


#——————————————————定义神经网络变量——————————————————
def lstm(X,batch_size):     
    print("-"*30 + ' lstm ' + '-'*30)
    # batch_size=tf.shape(X)[0]
    # time_step=tf.shape(X)[1]
    # print("batch_size = ",batch_size,"time_step = ",time_step)
    w_in=weights['in']
    b_in=biases['in'] 
    # print("lstm X = ",X.get_shape()) 
    my_input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    # print("my_input.shape = ",my_input.get_shape())
    input_rnn=tf.nn.tanh(tf.matmul(my_input,w_in)+b_in)
    # print("input_rnn.shape ----1 = ",input_rnn.get_shape())


    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    # print("input_rnn.shape ----2 = ",input_rnn.get_shape())
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=rnn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    # print("output.shape = ",output.get_shape())
    w_out=weights['out']
    b_out=biases['out']
    predict = tf.matmul(output,w_out)+b_out
    # print("predict.shape = ",predict.get_shape())
    print("-"*30 + ' lstm ' + '-'*30)
    return predict,final_states



#——————————————————训练模型——————————————————
def train_lstm(train_begin=2000,train_end=5800):
    batch_index,train_x,train_y=get_train_data(train_begin,train_end)
    print("train_x = ",len(train_x),len(train_x[0]),"train_y = ",len(train_y),len(train_y[0]))
    pred,_=lstm(X,batch_size = 60)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        #重复训练10000次
        try:
            saver.restore(sess,'./model2/model2.ckpt')
        except Exception as ex:
            print("[Exception Information] ",str(ex))
        for i in range(epoch):
            for step in range(len(batch_index)-1):
                x_batch = train_x[batch_index[step]:batch_index[step+1]]
                x_batch = np.array(x_batch)
                y_batch = train_y[batch_index[step]:batch_index[step+1]]
                y_batch = np.array(y_batch)
                print("x_batch = ",x_batch.shape,"y_batch = ",y_batch.shape)
                sess.run(train_op,feed_dict = {X: x_batch,Y: y_batch})
                _,loss_=sess.run([train_op,loss],feed_dict={X: x_batch,Y: y_batch})
                break
            break
            print("i = ",i,"loss_= ",loss_)
        sess.run(pred,feed_dict = {X: x_batch})

                
        # saver.save(sess,'./model2/model2.ckpt')


# train_lstm()


#————————————————预测模型————————————————————
def prediction():
    print("="*80)
    # X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    print("test data = ",np.array(test_x).shape,np.array(test_y).shape)
    predict_op,_=lstm(X,batch_size = 1)     
    
    with tf.Session() as sess:
        #参数恢复
        sess.run(tf.initialize_all_variables())
        saver=tf.train.Saver()
        saver.restore(sess,'./model2/model2.ckpt')
        test_predict=[]
        for step in range(len(test_x)-1):
          x_test = np.array([test_x[step]])
          # print("x_test.shape = ",x_test.shape)
          prob = sess.run(predict_op,feed_dict={X:x_test})   
          predict = prob.reshape((-1))
          print("predict.shape = ",predict.shape)
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',label = 'test_predict')
        plt.plot(list(range(len(test_y))), test_y,  color='r',label = 'test_y')
        plt.legend()
        plt.grid(True)
        plt.show()

prediction() 



def my_predict():
    pred,_=lstm(X,batch_size=2)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)

    mean,std,test_x,test_y=get_test_data(time_step)
    batch_index,train_x,train_y=get_train_data(0,5000)
    step = 2
    x_batch = train_x[batch_index[step]:batch_index[step+1]]
    x_batch = np.array(x_batch)
    x_batch = x_batch[0:2,:,:]
    print("x_batch.shape = ",x_batch.shape)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        #重复训练10000次
        saver.restore(sess,'./model2/model2.ckpt')
        x_test = np.array(test_x[0:2])
        print("x_test.shape = ",x_test.shape)
        prediction = sess.run(pred,feed_dict = {X: x_batch})
        print("prediction = ",prediction.shape)
# my_predict()