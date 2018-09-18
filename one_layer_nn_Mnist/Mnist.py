import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./Mnist_data', one_hot=True)
# print(mnist.train.images.shape, mnist.train.labels.shape)  # (55000, 784) (55000, 10)
# print(mnist.test.images.shape, mnist.test.labels.shape)  # (10000, 784) (10000, 10)
# print(mnist.validation.images.shape, mnist.validation.labels.shape)  # (5000, 784) (5000, 10)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-inout')

with tf.name_scope('variable'):
    W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1), name='Weight')
    b = tf.Variable(tf.zeros([10]), name='bias')
    # 生成日志信息
    tf.summary.histogram('weight', W)
    tf.summary.histogram('bias', b)

# 前向传播计算预测值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 计算损失函数
with tf.name_scope('loss_function'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    # 生产日志信息
    tf.summary.scalar('cross_entropy', cross_entropy)

# 训练迭代过程
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 计算精确度
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 生成日志信息
    tf.summary.scalar('accuracy', acc)

# 将所有的日志生成操作给合并
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # 定义一个write，并且利用writer写入Tensorflow计算图
    writer = tf.summary.FileWriter('./log', tf.get_default_graph())
    tf.global_variables_initializer().run()
    for i in range(2000):
        xs, ys = mnist.train.next_batch(100)
        summary, loss, _ = sess.run([merged, cross_entropy, train_step], feed_dict={x: xs, y_: ys})
        writer.add_summary(summary, i)
        if i % 100 == 0:
            print('epoch {},  training loss = {}'.format(i, loss))

    accuracy = sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('test accuracy = %2.f%%' % (accuracy*100))

writer.close()


