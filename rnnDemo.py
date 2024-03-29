import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# hyperparameters
lr = 0.001 # learning rate
training_iters = 100000 # 训练次数
batch_size = 128

n_inputs = 28 # MINST data input(img shape:28*28)
n_steps = 28 # time steps
n_hidden_unis = 128 # neurons in hidden layer
n_classes = 10 # MINST classes

# tf Graph input
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

# Define weights
weights = {
    # (28,128) 每个input对应128个hidden的神经元
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
    # (128,10) 每个神经元对应10个类比输出
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}
biases = {
    # (128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    # (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X,weight,biases):

    '''
    开始定义 RNN 主体结构, 这个 RNN 总共有 3 个组成部分
     ( input_layer, cell, output_layer).
     '''

    # hiden layer for input to cell
    ###############################
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X(128batch,28steps,28inputs)
    # ==>(128*28,28inputs)
    X = tf.reshape(X,[-1,n_inputs])
    # X_in = w*x + b
    # ==>(128batch*28steps,128hidden)
    X_in = tf.matmul(X,weights['in'])+biases['in']
    # ==>(128batch,28steps,128hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])

    # cell
    ###############################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    # lstm cell is divided into two parts (c_state,m_state)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state= init_state,time_major=False)


    # hidden layer for output as the final results
    ###############################
    results = tf.matmul(final_state[1],weights['out'])+biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1