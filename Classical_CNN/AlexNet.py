from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


# 打印tensor的尺寸
def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())


def inference(images):
    parameters = []

    print_activations(images)

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')

        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]  # 列表拼接

    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    with tf.name_scope('conv45') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    with tf.name_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)
        print_activations(pool3)

    with tf.name_scope('fc1') as scope:
        reshaped = tf.reshape(pool3, [batch_size, -1])
        dim = reshaped.shape[1].value
        fc1_weight = tf.Variable(tf.truncated_normal([dim, 4096], dtype=tf.float32, stddev=0.1), name='weight')
        fc1_bias = tf.Variable(tf.constant(0.0, shape=[4096]))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias, name=scope)
        print_activations(fc1)
        parameters += [fc1_weight, fc1_bias]

    with tf.name_scope('fc2') as scope:
        fc2_weight = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=0.1), name='weight')
        fc2_bias = tf.Variable(tf.constant(0.0, shape=[4096]))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weight) + fc2_bias, name=scope)
        print_activations(fc2)
        parameters += [fc2_weight, fc2_bias]

    with tf.name_scope('softmax') as scope:
        fc3_weight = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=0.1), name='weight')
        fc3_bias = tf.Variable(tf.constant(0.0, shape=[1000]))
        fc3 = tf.nn.softmax(tf.matmul(fc2, fc3_weight) + fc3_bias, name=scope)
        print_activations(fc3)
        parameters += [fc3_weight, fc3_bias]

    return fc3, parameters

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
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
        fc3, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, fc3, 'Forward')

        objective = tf.nn.l2_loss(fc3)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'Forward-backward')


if __name__ == '__main__':
    run_benchmark()