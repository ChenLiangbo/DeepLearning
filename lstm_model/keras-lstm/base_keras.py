#!usr/bin/env/python 
# -*- coding: utf-8 -*

# Sequential
model = Sequential()

# compile
model.compile(optimizer, loss, metrics=None, sample_weight_mode=None)
--- optimizer : str,optimizer='adam'
    # example
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    # sgd
    1,keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)


      lr：大于0的浮点数，学习率

      momentum：大于0的浮点数，动量参数

      decay：大于0的浮点数，每次更新后的学习率衰减值

      nesterov：布尔值，确定是否使用Nesterov动量

    2,RMSprop
    keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)


      lr：大于0的浮点数，学习率

      rho：大于0的浮点数

      epsilon：大于0的小浮点数，防止除0错误
    
    3,Adagrad
    keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)


      lr：大于0的浮点数，学习率

      epsilon：大于0的小浮点数，防止除0错误

    4,Adadelta


      lr：大于0的浮点数，学习率

      rho：大于0的浮点数

      epsilon：大于0的小浮点数，防止除0错误

---- loss : str  'mse'
    1,'mean_squared_error'或 'mse'
    2,'mean_absolute_error'或'mae'
    3,'mean_absolute_percentage_error'或'mape'
    4,'mean_squared_logarithmic_error'或'msle'
    5,'squared_hinge'
    5,'hinge'
    6,'binary_crossentropy'（亦称作对数损失，logloss）
    7,'categorical_crossentropy'
    8,'sparse_categorical_crossentrop'
    9,'kullback_leibler_divergence'
    10,'poisson'  (predictions - targets * log(predictions))
    11,'cosine_proximity'

---- metrics
    metrics：列表
    包含评估模型在训练和测试时的网络性能的指标
    典型用法是metrics=['accuracy']

---- sample_weight_mode
    如果你需要按时间步为样本赋权（2D权矩阵）
    将该值设为“temporal”。

model.fit(x, y, batch_size=32, #整数，指定进行梯度下降时每个batch包含的样本数
	            epochs=10,     #整数，训练的轮数，每个epoch会把训练集轮一遍
	            verbose=1,     #0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
	            callbacks=None,       #
	            validation_split=0.0, 
	            validation_data=None, 
	            shuffle=True, 
	            class_weight=None, 
	            sample_weight=None, 
	            initial_epoch=0)



    1,x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array

    2,y：标签，numpy array

    2,batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

    3,epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。

    4,verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

    5,callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数

    6,validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。

    7,validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。

    8,shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。

    9,class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）

    10,sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。

    11,initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。


model.evaluate( x, y, batch_size=32, verbose=1, sample_weight=None)

    x：输入数据，与fit一样，是numpy array或numpy array的list

    y：标签，numpy array

    batch_size：整数，含义同fit的同名参数

    verbose：含义同fit的同名参数，但只能取0或1

    sample_weight：numpy array，含义同fit的同名参数


model.predict( x, batch_size=32, verbose=0)
    
    函数的返回值是类别预测结果的numpy array或numpy


------------------------


model.predict_proba(x,batch_size = 32,verbose = 0)
    
    数按batch产生输入数据属于各个类别的概率

    函数的返回值是类别概率的numpy array


------------------------

model.train_on_batch(x, y, class_weight=None, sample_weight=None)
    
    在一个batch的数据上进行一次参数更新

    返回训练误差的标量值或标量值的list，与evaluate的情形相同


-------------------------------
model.test_on_batch(x, y, sample_weight=None)

    在一个batch的样本上对模型进行评估

    函数的返回与evaluate的情形相同


-----------------------------------
model.predict_on_batch(x)
    
    在一个batch的样本上对模型进行测试

    函数返回模型在一个batch上的预测结果

---------------------------------------
model.it_generator(generator, steps_per_epoch, 
	              epochs=1, 
	              verbose=1, 
	              callbacks=None, 
	              validation_data=None, v
	              alidation_steps=None, 
	              class_weight=None, 
	              max_q_size=10, 
	              workers=1, 
	              pickle_safe=False, 
	              initial_epoch=0)

    利用Python的生成器， 
    逐个生成数据的batch并进行训练。
    生成器与模型将并行执行以提高效率。
    例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练



generator：生成器函数，生成器的输出应该为：

    一个形如（inputs，targets）的tuple

    一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束

steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch

epochs：整数，数据迭代的轮数

verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

validation_data：具有以下三种形式之一

    生成验证集的生成器

    一个形如（inputs,targets）的tuple

    一个形如（inputs,targets，sample_weights）的tuple

validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数

class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。

sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。

workers：最大进程数

max_q_size：生成器队列的最大容量

pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。

initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。

============================================================
model
------------------------------------
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
---------------------------------------
model.compile(optimizer, loss, 
	          metrics=None, 
	          loss_weights=None, 
	          sample_weight_mode=None)
---------------------------------------

model.fit(x=None, y=None, 
	batch_size=32, 
	epochs=1, 
	verbose=1, 
	callbacks=None, 
	validation_split=0.0, 
	validation_data=None, 
	shuffle=True, 
	class_weight=None, 
	sample_weight=None, 
	initial_epoch=0)
--------------------------------------------

model.evaluate(x, y, 
	batch_size=32, 
	verbose=1, 
	sample_weight=None)

---------------------------------------------
model.predict(x, batch_size=32, verbose=0)
---------------------------------------------
model.train_on_batch(x, y, 
	class_weight=None, 
	sample_weight=None)
---------------------------------------------

layer
===========================================================================
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)

-------------------------------------------------
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
----------------------------------------------
---------------------------------------
该层不是共享层

    layer.input

    layer.output

    layer.input_shape

    layer.output_shape

--------------------------------------
有多个计算节点


    layer.get_input_at(node_index)

    layer.get_output_at(node_index)

    layer.get_input_shape_at(node_index)

    layer.get_output_shape_at(node_index)

----------------------------------------

========================================================
循环层Recurrent
keras.layers.recurrent.Recurrent(
	return_sequences=False, 
	go_backwards=False, 
	stateful=False, 
	unroll=False, 
	implementation=0)



return_sequences：布尔值，默认False，控制返回类型。
               若为True则返回整个序列，
               否则仅返回输出序列的最后一个输出

go_backwards：布尔值，默认为False，若为True，则逆向处理输入序列

stateful：布尔值，默认为False，
    若为True，则一个batch中下标为i的样本的最终状态将会用作下一个batch同样下标的样本的初始状态。

unroll：布尔值，默认为False，
    若为True，则循环层将被展开，否则就使用符号化的循环。当使用TensorFlow为后端时，循环网络本来就是展开的，因此该层不做任何事情。层展开会占用更多的内存，但会加速RNN的运算。层展开只适用于短序列。

implementation：0，1或2， 
         若为0，则RNN将以更少但是更大的矩阵乘法实现，因此在CPU上运行更快，但消耗更多的内存。
         如果设为1，则RNN将以更多但更小的矩阵乘法实现，因此在CPU上运行更慢，在GPU上运行更快，并且消耗更少的内存。
         如果设为2（仅LSTM和GRU可以设为2），则RNN将把输入门、遗忘门和输出门合并为单个矩阵，以获得更加在GPU上更加高效的实现。
         注意，RNN dropout必须在所有门上共享，并导致正则效果性能微弱降低。

input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)

input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接Flatten层，然后又要连接Dense层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果循环层不是网络的第一层，你需要在网络的第一层中指定序列的长度（通过input_shape指定）。

-----------------------------------------
LSTM(units, 
	activation='tanh', 
	recurrent_activation='hard_sigmoid', 
	use_bias=True, 
	kernel_initializer='glorot_uniform', 
	recurrent_initializer='orthogonal', 
	bias_initializer='zeros', 
	unit_forget_bias=True, 
	kernel_regularizer=None, 
	recurrent_regularizer=None, 
	bias_regularizer=None, 
	activity_regularizer=None, 
	kernel_constraint=None, 
	recurrent_constraint=None, 
	bias_constraint=None, 
	dropout=0.0, 
	recurrent_dropout=0.0)

Keras长短期记忆模型，关于此算法的详情，请参考本教程
参数

    units：输出维度

    activation：激活函数，为预定义的激活函数名（参考激活函数）

    recurrent_activation: 为循环步施加的激活函数（参考激活函数）

    use_bias: 布尔值，是否使用偏置项

    kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers

    recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers

    bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers

    kernel_regularizer：施加在权重上的正则项，为Regularizer对象

    bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象

    recurrent_regularizer：施加在循环核上的正则项，为Regularizer对象

    activity_regularizer：施加在输出上的正则项，为Regularizer对象

    kernel_constraints：施加在权重上的约束项，为Constraints对象

    recurrent_constraints：施加在循环核上的约束项，为Constraints对象

    bias_constraints：施加在偏置上的约束项，为Constraints对象

    dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

    recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
--------------------------------------------------