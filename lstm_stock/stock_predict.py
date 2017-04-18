#!usr/bin/env/python 
# -*- coding: utf-8 -*-

'''
Created on 2017年2月19日

@author: Lu.yipiao
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn 

# tensorflow 1.0.1
# from tensorflow.contrib import rnn

#——————————————————导入数据——————————————————————
f=open('./dataset/dataset_1.csv')  
df=pd.read_csv(f)     #读入股票数据
data=np.array(df['high'])   #获取最高价序列
data=data[::-1]      #反转，使数据按照日期先后顺序排列
#以折线图展示data
# plt.figure()
# plt.plot(data)
# plt.show()
normalize_data=(data-np.mean(data))/np.std(data)  #标准化
normalize_data=normalize_data[:,np.newaxis]       #增加维度
print("normalize_data = ",normalize_data.shape)

#生成训练集
#设置常量
time_step=50      #时间步
rnn_unit=15       #hidden layer units
batch_size=60     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
epoch = 10000     #训练周期

test_number = 200
train_number = normalize_data.shape[0] - test_number
test_y = normalize_data[train_number:]

train_x,train_y=[],[]   #训练集
for i in range(len(normalize_data)-time_step-1 - test_number):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 
print("x,y = ",x.shape,y.shape)
print("train_x = ",len(train_x),len(train_x[0]),len(train_y))

#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }



#——————————————————定义神经网络变量——————————————————
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input1=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.nn.tanh(tf.matmul(input1,w_in)+b_in)
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入

    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # cell=rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    # output_rnn,final_states=tf.nn.rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    # output_rnn,final_states=rnn.static_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = rnn.dynamic_rnn(cell, input_rnn, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #重复训练10000次
        for i in range(epoch):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                x_batch = np.array(train_x[start:end])
                y_batch = np.array(train_y[start:end])
                # print("x_batch = ",x_batch.shape,"y_batch = ",y_batch.shape)
                _,loss_=sess.run([train_op,loss],feed_dict={X:x_batch,Y:y_batch})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                # break
                if step%300==0:
                    print("i = ",i,step,loss_)
                    # print("保存模型：",saver.save(sess,'stock.model'))
                step+=1

        saver = tf.train.Saver()
        saver.save(sess,'./model1/model1.ckpt')
        sess.close()


# train_lstm()


#————————————————预测模型————————————————————
def prediction():
    pred,_=lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
    saver=tf.train.Saver()
    with tf.Session() as sess:
        #参数恢复
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,'./model1/model1.ckpt')
        #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        print("prev_seq.shape = ",np.array(prev_seq).shape)
        predict=[]
        #得到之后100个预测结果
        for i in range(test_number):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            # print("next_seq[-1] = ",next_seq[-1])
            predict.append(next_seq[-1])
            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
            # print("prev_seq = ",prev_seq.shape)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        # plt.plot(list(range(test_number)),test_y.tolist(),'ro',label = 'y_real')
        # plt.plot(list(range(test_number)),test_y.tolist(),'r-')
        # plt.plot(list(range(len(predict))),predict,'bo',label = 'y_predict')
        # plt.plot(list(range(len(predict))),predict,'b--')
        plt.grid(True)
        plt.legend()

        plt.show()

prediction() 
