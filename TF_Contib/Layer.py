import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

height, width = 3, 3
images = np.random.uniform(size=(5, height, width, 3))
output = layers.avg_pool2d(images, [3, 3], stride=2, padding='VALID')
print(output.shape)
print(output.op.name)
output = layers.conv2d(images, num_outputs=32, stride=1, kernel_size=[3, 3], padding='SAME')
print(output.shape)
print(output.op.name)

weights = tf.contrib.framework.get_variables_by_name('weights')[0]
weights_shape = weights.get_shape().as_list()
images = tf.random_uniform((5, height, width, 32), seed=1)

with tf.contrib.framework.arg_scope(
        [layers.conv2d],
        normalizer_fn=layers.batch_norm,
        normalizer_params={'decay': 0.9}
):
    net = layers.conv2d(images, 32, [3, 3])
    net = layers.conv2d(net, 32, [3, 3])
print(net.shape)
print(net.op.name)


height, width = 3, 3
inputs = np.random.uniform(size=(5, height*width*3))
with tf.name_scope('fe'):
    fc = layers.fully_connected(inputs=inputs, num_outputs=7, outputs_collections='outputs', scope='fc')
output_collected = tf.get_collection('outputs')[0]
print(output_collected.shape)
print(output_collected.op.name)
print(fc.shape)
print(fc.op.name)

x = np.random.uniform(size=(5, height, width, 3))

# 等价操作
y = layers.repeat(x, 3, layers.conv2d, 64, [3, 3], scope='conv1')
print(y.shape)
print(y.op.name)

# x = layers.conv2d(x, 64, [3, 3], scope='conv1/conv1_1')
# x = layers.conv2d(x, 64, [3, 3], scope='conv1/conv1_2')
# y = layers.conv2d(x, 64, [3, 3], scope='conv1/conv1_3')
# print(y.shape)

# x = np.random.uniform(size=(5, height*width*3))
# y = layers.stack(x ,layers.fully_connected, [32, 64, 128], scope='fc')
# print(y.shape)
# print(y.op.name)

x = layers.fully_connected(x, 32, scope='fc/fc_1')
x = layers.fully_connected(x, 64, scope='fc/fc_2')
y = layers.fully_connected(x, 128, scope='fc/fc_3')
print(y.shape)
print(y.op.name)
0