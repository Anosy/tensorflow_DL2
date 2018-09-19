import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform ((fan_in, fan_out),
        minval = low, maxval = high, dtype = tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input  # 输入变量数
        self.n_hidden = n_hidden  # 隐藏层节点的数量
        self.transfer = transfer_function  # 隐藏层的激活函数
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale  # 高斯噪声
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义网络结构
        # 定义输入
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 定义隐藏层，并且在输入数据中加入噪声
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']),
                                           self.weights['b1']))
        # 定义输出层，最后输出层不需要激活函数
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 定义自编码器的损失函数
        # mse损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)


    # 初始化参数
    def _initialize_weights(self):
        all_weights = {}
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 定义执行损失函数和训练
    def partial_fit(self, X):
        cost, opt = self.sess.run([self.cost, self.optimizer],
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 只计算损失函数
    def cale_total_coat(self, x):
        return self.sess.run(self.cost, feed_dict={self.x: x, self.scale: self.training_scale})

    # 返回自编码器隐含层的输出结果
    def transform(self, x):
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.scale: self.training_scale})

    # 将隐含层的结果作为输入，通过重建层来提取高阶的特征来复原原始的数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 复原过程
    def reconstruct(self, x):
        return self.sess.run(self.reconstruct, feed_dict={self.x: x, self.scale: self.training_scale})

    # 获取隐含层权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层的偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 载入mnist数据，作为实验数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 将训练集和测试集的数据给标准化
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_test, X_train


# 选取随机batch
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 得到标准化后的结果
X_tarin, X_test = standard_scale(mnist.train.images, mnist.train.images)

# 模型参数
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

for epoch in range(training_epochs):
    avg_coat = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_tarin, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_coat += cost / n_samples

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", '{:.9f}'.format(avg_coat))
print("Total coat:" + str(autoencoder.cale_total_coat(X_test / int(mnist.test.num_example))))
