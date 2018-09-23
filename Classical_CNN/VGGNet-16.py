from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    '''
    创建卷积层，并且将本层的参数给存入到参数列表中
    :param input_op: 输入的tensor
    :param name: 该层的名字
    :param kh: 卷积核的高
    :param kw: 卷积核的宽
    :param n_out: 卷积核输出的通道数
    :param dh: 步长的高
    :param dw: 步长的宽
    :param p: 参数列表
    :return:
    '''
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable(scope + 'b', shape=[n_out], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_op, kernel, [1, dh, dw, 1], padding='SAME')
    z = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(z, name=scope)

    p += [kernel, biases]
    return activation


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(scope+'b', shape=[n_out], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
    activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
    p += [kernel, biases]
    return activation


def maxpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


def inference_op(input_op, keep_prob):
    p = []
    # 1st layer
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = maxpool_op(conv1_2, name='poo11', kh=2, kw=2, dw=2, dh=2)
    print_activations(pool1)
    # 2nd
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = maxpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)
    print_activations(pool2)
    # 3rd
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = maxpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)
    print_activations(pool3)
    # 4th
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = maxpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)
    print_activations(pool4)
    # 5th
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = maxpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)
    print_activations(pool5)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')
    print_activations(resh1)
    # 6th
    fc6 = fc_op(resh1, name='fc6', n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    print_activations(fc6_drop)

    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')
    print_activations(fc7_drop)

    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
    print_activations(fc8)

    softmax = tf.nn.softmax(fc8, name='softmax')
    print_activations(softmax)

    predictions = tf.argmax(softmax, 1)

    return predictions, softmax, fc8, p


def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if i % 10 == 0:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.2f sec /batch' % (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],dtype=tf.float32, stddev=0.1))
        keep_prob = tf.placeholder(tf.float32)
        prediction, softmax, fc8, parameters = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, prediction, {keep_prob: 1.0}, 'Forward')

        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, 'Forward-backward')

if __name__ == '__main__':
    run_benchmark()