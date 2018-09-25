import tensorflow as tf
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify' + filename + '.Can you get to it with a browser?'
        )
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        # f.namelist 返回所有文件名列表， tf.compat.as_str 将数据转化成单词的列表
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print(words[:5])
print('Data Size', len(words))

# 取前50000个词作为vocabulary
vocabulary_size = 50000


def build_dataset(words):

    '''
    创建词汇表，
    :param words: 所有的词汇
    :return:
    data: 词在词汇表的位置
    count: 每个单词以及其的词频
    dictionary: 词汇表
    reverse_dictionary: 反转词汇表

    '''

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    '''
    生成用来训练的batch数据
    :param batch_size: batch的大小
    :param num_skips:  每个单词生成样本数量
    :param skip_window: 单词最远可以联系的距离，如果设为1表示只能核紧邻的两个单词生成样本
    :return:
    '''
    global data_index
    assert batch_size % num_skips == 0  # 确保一个batch至少包含一个词汇的所有的样本
    assert num_skips <= 2 * skip_window  # 生成的样本是数量不能大于skip_window两倍
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # 每个单词创建样本所需要用到的单词数量
    buffer = collections.deque(maxlen=span)  # 创建最大容量为span的队列

    for _ in range(span):  # 把span个单词读入buffer作为初始值
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):  # 需要生成样本的单词的数量，循环中每次循环对一个目标单词生成样本
        target = skip_window
        targets_to_avoid = [skip_window]  # 需要避免的单词表
        for j in range(num_skips):  # 每个单词生成的样本数量， 循环中每次循环对当前语境单词生成一个样本
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])  # buffer存在固定的长度，所以加入一个样本之后，将会使第一个样本被剔除
        data_index = (data_index + 1) % len(data)

    return batch, labels


# # test code
# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 从100个数中随机抽取16个数
num_sampled = 64  # 负样本的噪声数量

graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        # 随机生成所有单词的词向量
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 查找输入train_inputs对应的向量embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=train_labels,
        inputs=embed,
        num_sampled=num_sampled,
        num_classes=vocabulary_size
    ))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 计算嵌入向量embeddings的L2范数
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    # 查找验证单词的嵌入向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 计算验证单词的嵌入向量与词汇表中所有单词的相似度
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 100001
with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        bath_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: bath_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("average loss at step:", step, ":", average_loss)
            average_loss = 0
        if step % 10000 == 0:
            print(similarity.shape)
            sim = similarity.eval()
            print(sim.shape)
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = "nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# 可视化word2vec效果的函数
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "more labels than embeddings"
    plt.figure(figsize=(18, 81))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


# 实现降维
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

