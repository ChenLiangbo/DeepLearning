#!usr/bin/env/python 
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import tensorflow as tf

#定义常量
rnn_unit=10       #hidden layer units
input_size=5      
output_size=1
lr=0.0006         #学习率

filename = '../data/yahoo_finance5.xls'
xls = pd.ExcelFile(filename)
df = xls.parse('Sheet1', index_col='Date') # train

data=df.iloc[:,0:5].values   #取第3-10列
print("data = ",len(data))

#——————————获取训练集——————————
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:4]
       y=normalized_train_data[i:i+time_step,4,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#——————————获取测试集——————————
def get_test_data(time_step=20,test_begin=700):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:4]
       y=normalized_test_data[i*time_step:(i+1)*time_step,4]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:4]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,4]).tolist())
    return mean,std,test_x,test_y

#——————————————————定义神经网络变量——————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#——————————————————训练模型——————————————————
def train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=5800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    module_file = tf.train.latest_checkpoint()    
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(2000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 200==0:
                print("保存模型：",saver.save(sess,'stock2.model',global_step=i))



#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint()
        saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]) #acc为测试集偏差



if __name__ == '__main__':
    batch_index,train_x,train_y = get_train_data()
    print("train data = ",len(train_x),len(train_x),len(train_y))
    mean,std,test_x,test_y = get_test_data()
    print("mean = ",mean)
    print("test data = ",len(test_x),len(test_x),len(test_y))
    train_lstm()
    prediction()