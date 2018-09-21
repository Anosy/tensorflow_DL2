from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('./Mnist_data', one_hot=True)

input_node = 784
output = 10
image_size = 28
num_channels = 1
num_labels = 10

# 第一层卷积的尺寸和参数
conv1_deep = 32
conv1_size = 5
# 第二层卷积层的尺寸和深度
conv2_deep = 64
conv2_size = 5
# 全连接层的节点个数
fc_size = 512


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, image_size* image_size*num_channels], 'x-input')
    x_image = tf.reshape(x, [-1, 28, 28, 1], 'x_image')
    y_ = tf.placeholder(tf.float32, [None, num_labels], 'y-input')
    keep_prob = tf.placeholder(tf.float32)

    tf.summary.image('input', x_image, 2)

with tf.variable_scope('layer1-conv1'):
    W_conv1 = tf.get_variable('weight', [conv1_size, conv1_size, num_channels, conv1_deep],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable('bias', [conv1_deep], initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    relu1 = tf.nn.relu(conv1)

    tf.summary.image('conv1', tf.expand_dims(conv1[:, : ,: ,0], -1), 2)
    tf.summary.image('relu1', tf.expand_dims(relu1[:, : ,: ,0], -1), 2)

with tf.variable_scope('layer2-max_pool1'):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # tf.summary.image('pool1', pool1)

with tf.variable_scope('layer3-conv2'):
    W_conv2 = tf.get_variable('weight', [conv2_size, conv2_size, conv1_deep, conv2_deep],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable('bias', [conv2_deep], initializer=tf.constant_initializer(0.0))

    conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
    relu2 = tf.nn.relu(conv2)

    tf.summary.image('conv2', tf.expand_dims(conv2[:, : ,: ,0], -1), 2)
    tf.summary.image('relu2', tf.expand_dims(relu2[:, : ,: ,0], -1), 2)

with tf.variable_scope('layer4-max_pool2'):
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # tf.summary.image('pool2', pool2)

with tf.variable_scope('layer5-fc1'):
    W_fc1 = tf.get_variable('weight', [7 * 7 * conv2_deep, fc_size],
                            initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable('bias', [fc_size], initializer=tf.constant_initializer(0.0))
    pool2_flat = tf.reshape(pool2, [-1, 7*7*conv2_deep])
    fc1 = tf.matmul(pool2_flat, W_fc1) + b_fc1
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)

with tf.variable_scope('layer6-fc2'):
    W_fc2 = tf.get_variable('weight', [fc_size, num_labels], initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable('bias', [num_labels], initializer=tf.constant_initializer(0.0))

    fc2 = tf.matmul(fc1_dropout, W_fc2) + b_fc2
    y = tf.nn.softmax(fc2)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits (logits=fc2, labels=y_,
                                                                              name='cross_entropy'))
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('Acc'):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', acc)

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('./log', tf.get_default_graph())

    for i in range(1, 2001):
        xs, ys = mnist.train.next_batch(50)


        if i % 100 == 0:
            train_accuracy = sess.run(acc, feed_dict={x: xs, y_: ys, keep_prob: 1.0})
            print('epoch % d, training accuracy= % g' % (i, train_accuracy))

        loss, _, summary = sess.run([cross_entropy, train_step, merged],
                                    feed_dict={x: xs, y_: ys, keep_prob: 0.5})
        writer.add_summary(summary, global_step=i)
        if i % 100 == 0:
            print('epoch % d, training loss= % g' % (i, loss))
            
    writer.close()
