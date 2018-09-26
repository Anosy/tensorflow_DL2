import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn, metrics, layers
from sklearn.datasets import load_iris
from sklearn import model_selection

tf.logging.set_verbosity(tf.logging.INFO)

iris = load_iris()
X_train, X_test, y_train,y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2,
                                                                   random_state=35)
# validation_metrics = {
#     'accuracy': metrics.streaming_accuracy,
#     'precision': metrics.streaming_precision,
#     'recall': metrics.streaming_recall,
# }

validation_monitor = learn.monitors.ValidationMonitor(
    X_train,
    y_train,
    every_n_steps=50,  # 每50步执行一次监视
    early_stopping_metric='loss',
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200  # 超过200步数，损失不减少，停止迭代
)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 15, 10],
    model_dir='./iris_model_dir',
    config=learn.RunConfig(save_checkpoints_steps=2)
)

classifier.fit(x=X_train, y=y_train, steps=100, monitors=[validation_monitor])

accuracy_score = classifier.evaluate(x=X_test,  y=y_test)['accuracy']
print(accuracy_score)

