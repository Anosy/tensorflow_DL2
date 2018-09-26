import tensorflow as tf
from tensorflow.contrib import learn, layers, metrics, framework
from sklearn.model_selection import train_test_split
from sklearn import datasets

def _input_fn(num_epochs=None):
    features = {
        'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]), num_epochs=num_epochs),
        'language': tf.SparseTensor(values=['en', 'fr', 'ch'], indices=[[0, 0], [0, 1], [2, 0]], dense_shape=[3, 2])
    }
    return features, tf.constant([[1], [0], [0]], dtype=tf.int32)


language_column = layers.sparse_column_with_hash_bucket('language', hash_bucket_size=20)
feature_columns = [
    layers.embedding_column(language_column, dimension=1),
    layers.real_valued_column('age')
]

classifier = learn.DNNClassifier(
    n_classes=2,
    feature_columns=feature_columns,
    hidden_units=[3, 3],
    config=learn.RunConfig(tf_random_seed=1)
)

classifier.fit(input_fn=_input_fn, steps=10)
scores = classifier.evaluate(input_fn=_input_fn, steps=1)
print(scores)


'''------------------Part two-------------------'''


# # 定义每行数据具有不同的权重
# def _input_fn_train():
#     target = tf.constant([[1], [0], [0], [0]])
#     features = {
#         'x': tf.ones(shape=[4, 1], dtype=tf.float32),
#         'w': tf.constant([[100.], [3.], [2.], [2.]])
#     }
#     return features, target
#
# classifier = learn.DNNClassifier(
#     weight_column_name='w',
#     feature_columns=[layers.real_valued_column('x')],
#     hidden_units=[3, 3],
#     config=learn.RunConfig(tf_random_seed=1)
# )
#
# classifier.fit(input_fn=_input_fn_train, steps=10)
# scores = classifier.evaluate(input_fn=_input_fn_train, steps=1)
# print(scores)

'''------------------Part three-------------------'''


# # 自己定义metric函数
# def _input_fn_train():
#     target = tf.constant([[1], [0], [0], [0]])
#     features = {
#         'x': tf.ones(shape=[4, 1], dtype=tf.float32),
#     }
#     return features, target
#
#
# def _my_metric_op(predictions, targets):
#     predictions = tf.slice(predictions, [0, 1], [-1, 1])
#     return tf.reduce_sum(tf.multiply(predictions, targets))
#
#
# classifier = learn.DNNClassifier(
#     feature_columns=[layers.real_valued_column('x')],
#     hidden_units=[3, 3],
#     config=learn.RunConfig(tf_random_seed=1)
# )
#
# classifier.fit(input_fn=_input_fn_train, steps=10)
#
# scores = classifier.evaluate(
#     input_fn=_input_fn_train,
#     steps=10,
#     metrics={
#         'my_accuracy': metrics.streaming_accuracy,
#         ('my_prediction', 'classes'): metrics.streaming_precision,
#         ('my_metric', 'probabilities'): _my_metric_op
#     }
# )
# print(scores)


'''------------------Part three-------------------'''


# def optimizer_exp_decay():
#     global_step = framework.get_or_create_global_step(),
#     learning_rate = tf.train.exponential_decay(
#         learning_rate=0.1, global_step=global_step,
#         decay_steps=100, decay_rate=0.001
#     )
#
#     return tf.train.AdagradOptimizer(learning_rate=learning_rate)
#
#
# iris = datasets.load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
#
# feature_columns = learn.infer_real_valued_columns_from_input(X_train)
# classifier = learn.DNNClassifier(
#     feature_columns=feature_columns,
#     hidden_units=[10, 20, 10],
#     n_classes=3,
#     optimizer=optimizer_exp_decay
# )
#
# classifier.fit(X_train, y_train, steps=10)
# scores = classifier.evaluate(X_test, y_test, steps=1)
# print(scores)
