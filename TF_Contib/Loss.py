import tensorflow as tf
from tensorflow.contrib import layers, losses

predictions = tf.constant([4, 5, 12, 8, 1, 3], shape=[2, 3])
targets = tf.constant([1, 9, 2, -5, -2, 6], shape=[2, 3])
weight = tf.constant([1.2, 0.8], shape=[2, ])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
loss = losses.absolute_difference(predictions, targets, weight)
print(loss.eval(session=sess))



predictions = tf.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
labels = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

loss = losses.softmax_cross_entropy(predictions, labels)
print(loss.eval(session=sess))
