import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from sklearn import datasets, model_selection, metrics


def my_model(features, target):
    # 对target进行读热编码， 维度为3
    target = tf.one_hot(target, 3, 1, 0)
    # 堆叠多层的全连接神经网络，隐含层的层数分别为10， 20， 10， 并且通过不同层的训练得到新的feature
    features = layers.stack(features, layers.fully_connected, [10, 20, 10])
    # 给网络添加一个初始权重为0的逻辑回归层
    prediction, loss = learn.models.logistic_regression_zero_init(features, target)
    # 定义学习方法和学习率
    train_op = layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
                                    learning_rate=0.1)

    return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


if __name__ == '__main__':
    iris = datasets.load_iris()
    X_train, X_test, y_train,y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2,
                                                                       random_state=35)
    # 自定义模型 输入(feature, targets) 输出 (predictions, loss, train_op)
    classifier = learn.Estimator(model_fn=my_model)
    classifier.fit(X_train, y_train, steps=10)
    predictions = classifier.predict(X_test)
    predictions_label = [x['class'] for x in predictions]
    acc = metrics.accuracy_score(predictions_label, y_test)
    print('Acc: {}'. format(acc))
