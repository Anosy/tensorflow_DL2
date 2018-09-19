import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Mnist_data', one_hot=True)

n_input = 784
n_hidden1 = 300
n_output = 10
batch_size = 100
keep_prob = 0.75
training_step = 3000


with tf.name_scope('layer1'):
    w1 = tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1), name='weight1')
    b1 = tf.Variable(tf.zeros([n_hidden1]), 'bias1')
    tf.summary.histogram('w1', w1)

with tf.name_scope('layer2'):
    w2 = tf.Variable(tf.truncated_normal([n_hidden1, n_output], stddev=0.1), name='weight2')
    b2 = tf.Variable(tf.zeros([n_output]), 'bias2')
    tf.summary.histogram('w2', w2)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_input], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, n_output], name='y-input')

layer1_active_output = tf.nn.relu(tf.matmul(x, w1) + b1)
layer1_output = tf.nn.dropout(layer1_active_output, keep_prob)
layer2_output = tf.matmul(layer1_output, w2) + b2
layer2_active_output = tf.nn.softmax(layer2_output)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=layer2_output),
                                   name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


with tf.name_scope('Acc'):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(layer2_active_output, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', acc)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    writer = tf.summary.FileWriter('./log', tf.get_default_graph())

    for i in range(training_step):
        xs, ys = mnist.train.next_batch(batch_size)
        loss, _, summary = sess.run([cross_entropy, train_step, merged], feed_dict={x: xs, y_: ys})
        writer.add_summary(summary, i)
        if i % 100 == 0:
            print('epoch {}, loss = {}'.format(i, loss))
    accuracy = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('epoch {}, test accuracy = {}'.format(training_step, accuracy))

    writer.close()
