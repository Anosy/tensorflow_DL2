import tensorflow as tf
import time
import numpy as np
import cifar10
import cifar10_input

max_steps = 5000
batch_size = 128
data_dir = './cifar-10-batches-bin'

def variable_with_weight_loss(shape, stdevv, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stdevv))
    if w1 is not None:  # 这里的w1表示L2正则化前面的系数
        weigt_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weigt_loss)
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(w1)(var))  # 等价表示
    return var

# 加载数据
cifar10.maybe_download_and_extract()
# 训练集
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
print(labels_train.shape)
# 测试集
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
# 载入数据
images_in = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_in = tf.placeholder(tf.int32, [batch_size])
# 第一个卷积层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stdevv=0.05, w1=0.0)  # 这里不使用L2正则化
kernel1 = tf.nn.conv2d(images_in, weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/ 9.0, beta=0.75)  # LRN局部响应归一化
# 第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5 ,64, 64], stdevv=0.05, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1 ,1 ,1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/ 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 全连接层1
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stdevv=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
# 全连接层2
weight4 = variable_with_weight_loss(shape=[384, 192], stdevv=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
# 最后一层
weight5 = variable_with_weight_loss(shape=[192, 10], stdevv=1/199.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.matmul(local4, weight5) + bias5

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

loss = loss(logits, label_in)
train_op = tf.train.AdamOptimizer(0.05).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_in, 1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners() #启动线程，在图像数据增强队列例使用了16个线程进行加速。

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    free, loss_value = sess.run([train_op, loss], feed_dict = {images_in: image_batch, label_in: label_batch})
    duration = time.time() - start_time #运行时间
    if step %10 == 0:
        example_per_sec = batch_size/duration#每秒训练样本数
        sec_per_batch = float(duration) #每个batch时间
        format_str = ('step %d, loss=%.2f(%.1f exaples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, example_per_sec, sec_per_batch))

num_examples = 1000
import math
num_iter = int(math.ceil(num_examples / batch_size))#math.ceil()为向上取整
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op],feed_dict={images_in: image_batch, label_in: label_batch})
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
