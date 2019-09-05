# -*- coding: utf-8 -*-
"""
TensorFlow 手写数字识别
    初始化计算图
    获取训练数据集和测试数据集
    将原始训练数据集分割为验证数据集和新的训练数据集
    创建占位符
    创建变量
    为计算图创建模型操作
    创建损失函数
    创建优化器操作
    创建 Tensorboard 汇总
    创建模型评估操作
    初始化模型
    训练模型
    输出最终模型的测试误差
    启动并查看 Tensorboard
WangKai 编写于 2019/04/30 13:00:00 (UTC+08:00)
  中国科学院大学, 北京, 中国
  地球与行星科学学院
  Comments, bug reports and questions, please send to:
  wangkai185@mails.ucas.edu.cn
Versions:
  最近更新: 2019/04/30
      算法构建，测试
"""
import os
import sys
import time
import gzip
import numpy as np
import tensorflow as tf
from six.moves import urllib
from datetime import datetime
from tensorflow.python.framework import ops

""" 初始化计算图 """
# 重置计算图
ops.reset_default_graph()
# 创建图会话
sess = tf.Session()

""" 获取训练数据集和测试数据集 """
print('Import datasets:')
# 定义数据下载链接和数据存放路径
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATASETS_DIR = 'MNIST_data'
# 定义手写数字图片的长宽、通道数和像素值范围
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255


# 定义获取数据集的函数
def get_datasets(filename, set_type, num_images):
    # 确保数据存放的文件夹存在
    if not tf.gfile.Exists(DATASETS_DIR):
        tf.gfile.MakeDirs(DATASETS_DIR)
    # 拼接文件路径
    filepath = os.path.join(DATASETS_DIR, filename)
    # 如果数据集不存在则通过下载链接下载数据集
    if not tf.gfile.Exists(filepath):
        print('Downloading datasets: ' + filename + ' from ' + SOURCE_URL + filename)
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('  Successfully downloaded datasets: ' + filename + ', size =  ' + str(size) + ' bytes.')
    else:
        print('Datasets ' + filename + ' already downloaded.')
    # 加载手写数字图像数据集
    if set_type == 'data':
        print('  Extracting ' + filepath)
        with gzip.open(filepath) as bytestream:
            # 每个像素存储在文件中的大小为 16 bits
            bytestream.read(16)
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            # 调整像素值 [0, 255] 到 [-0.5, 0.5]
            data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            # 调整为 4 维张量 [image_index, y, x, channels]
            data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            return data
    # 加载手写数字标签数据集
    elif set_type == 'labels':
        print('  Extracting ' + filepath)
        with gzip.open(filepath) as bytestream:
            # 每个标签存储在文件中的大小为 8 bits
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            return labels


# 获取数据集
train_data = get_datasets('train-images-idx3-ubyte.gz', 'data', 60000)
train_labels = get_datasets('train-labels-idx1-ubyte.gz', 'labels', 60000)
test_data = get_datasets('t10k-images-idx3-ubyte.gz', 'data', 10000)
test_labels = get_datasets('t10k-labels-idx1-ubyte.gz', 'labels', 10000)

""" 将原始训练数据集分割为验证数据集和新的训练数据集 """
# 定义验证数据集大小
VALIDATION_SIZE = 5000
# 将原始训练数据集分割为验证数据集和新的训练数据集
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]

""" 创建占位符 """
# 声明批量训练集大小和当前批量训练集中模型评估训练集大小
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
# 创建占位符，作为一批训练数据的输入节点
images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
labels_placeholder = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
eval_placeholder = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

""" 创建变量 """
# 声明手写数字标签数量
NUM_LABELS = 10
# 创建变量，包含所有可训练的权重
conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, dtype=tf.float32),
                            name='Conv1_weights')
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32), name='Conv1_biases')
conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32), name='Conv2_weights')
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='Conv2_biases')
fc1_weights = tf.Variable(
    tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, dtype=tf.float32),
    name='Fc1_weights')
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='Fc1_biases')
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, dtype=tf.float32), name='Fc2_weights')
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32), name='Fc2_biases')

""" 为计算图创建模型操作 """


def model(data, train=False):
    # 2D 卷积神经网络(Convolutional Neural Networks, CNN)
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 使用整流线性单元(Rectifier linear unit，ReLU)激励函数
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # 使用最大池化(max-pooling)方法，保留更多的纹理信息。均值池化(mean-pooling)则保留更多的图像背景信息
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第二层，使用相同的方法
    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 将特征图变换为 2D 矩阵，以将其提供给完全连接的图层
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # 全连接层
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # 在训练时，添加 dropout 层
    if train:
        hidden = tf.nn.dropout(hidden, 0.5)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


""" 创建损失函数 """
# 模型输出
model_output = model(images_placeholder, True)
# 样本损失
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=model_output),
                      name='Loss')
# L2 正则化损失
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(
    fc2_biases))
# 总损失 = 样本损失 + L2 正则化损失
loss += 5e-4 * regularizers

""" 创建优化器操作 """
# 声明训练样本集大小
train_size = train_labels.shape[0]
# 创建变量，用以控制学习率衰减
learning_rate_batch = tf.Variable(0, dtype=tf.float32)
# 学习率是从 0.01 开始的指数衰减
learning_rate = tf.train.exponential_decay(0.01, learning_rate_batch * BATCH_SIZE, train_size, 0.95, staircase=True,
                                           name='Learning_rate')
# 用 momentum 优化器优化，最小化损失
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=learning_rate_batch)

""" 创建 Tensorboard 汇总 """
# 创建时间戳，并创建当前 Tensorboard 日志文件夹
time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
log_dir = 'tensorboard/mnist_' + time_stamp
# 创建 summary_writer，将 Tensorboard summary 写入到当前日志文件夹
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
# 确保 summary_writer 写入的文件夹存在
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# 创建 Tensorboard 操作汇总标量
with tf.name_scope('Loss_and_Learning_rate'):
    tf.summary.scalar('Loss', tf.squeeze(loss))
    tf.summary.scalar('Learning_rate', tf.squeeze(learning_rate))
# 添加 Tensorboard 直方图汇总张量
with tf.name_scope('Variables'):
    tf.summary.histogram('Conv1_biases', conv1_biases)
    tf.summary.histogram('Conv2_biases', conv2_biases)
    tf.summary.histogram('Fc1_biases', fc1_biases)
    tf.summary.histogram('Fc2_biases', fc2_biases)
# 创建完这些汇总操作之后，创建汇总合并操作综合所有的汇总数据
summary = tf.summary.merge_all()

""" 创建模型评估操作 """
# 对当前训练集进行小批量预测
train_prediction = tf.nn.softmax(model_output)
# 评估小批量预测的结果
eval_prediction = tf.nn.softmax(model(eval_placeholder))


# 模型评估函数，对当前训练集的模型进行评估
def eval_in_batches(data, sess):
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    # 多批次评估
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={eval_placeholder: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(eval_prediction, feed_dict={eval_placeholder: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


""" 初始化模型 """
# 初始化模型变量
init = tf.global_variables_initializer()
# 启动图
sess.run(init)
print('\nInitialized model.')

""" 训练模型 """
# 声明总的训练代数（将训练集训练多少次）和评估模型的频率（每隔多少次评估一次模型）
NUM_EPOCHS = 3
EVAL_FREQUENCY = 100
# 计算总的迭代次数
generations = int(NUM_EPOCHS * train_size) // BATCH_SIZE
# 获取当前系统时间
start_time = time.time()
# 通过迭代训练模型
print('Start the training...')
for step in range(generations):
    # 计算当前批量数据集在总训练集中的偏移量
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    # 生成当前批量训练数据集
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # 生成要送入计算图的数据的字典
    feed_dict = {images_placeholder: batch_data, labels_placeholder: batch_labels}
    # 运行优化器进行参数更新
    sess.run(optimizer, feed_dict=feed_dict)
    # 在指定的评估频率上打印模型的评估信息
    if (step + 1) % EVAL_FREQUENCY == 0 or (step + 1) == generations:
        # 运行计算图得到评估信息
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
        # 计算当前所用时间
        elapsed_time = time.time() - start_time
        # 重置开始时间
        start_time = time.time()
        # 小批量评估模型在验证数据集上的预测结果
        predictions_eval = eval_in_batches(validation_data, sess)
        # 计算当前批量数据集的误差和验证数据集的小批量评估误差
        minibatch_error = 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == batch_labels) / predictions.shape[0])
        validation_error = 100.0 - (
                    100.0 * np.sum(np.argmax(predictions_eval, 1) == validation_labels) / predictions_eval.shape[0])
        # 更新汇总数据
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step + 1)
        summary_writer.flush()
        # 打印当前模型评估信息，输出时间为每次迭代平均用时
        print('\nStep #%d of %d (epoch %.2f of %d): Average time = %.1f ms' %
              (step + 1, generations, float(step) * BATCH_SIZE / train_size, NUM_EPOCHS,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f; learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % minibatch_error)
        print('Validation error: %.1f%%' % validation_error)
        sys.stdout.flush()
print('\nEnd of the training.')

""" 输出最终模型的测试误差 """
# 小批量评估模型在测试数据集上的预测结果
predictions_eval = eval_in_batches(test_data, sess)
# 计算测试数据集的小批量评估误差
test_error = 100.0 - (100.0 * np.sum(np.argmax(predictions_eval, 1) == test_labels) / predictions_eval.shape[0])
print('Test error: %.1f%%' % test_error)

""" 启动并查看 Tensorboard """
# 启动 Tensorboard，在浏览器中输入 http://localhost:6006 即可打开 Tensorboard 面板，
# 在命令行按下 Ctrl+C 则关闭 Tensorboard。
print('\nOpening the tensorboard...')
os.system("tensorboard --logdir " + log_dir + " --host localhost --port 6006")
