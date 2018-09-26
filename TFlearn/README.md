# 本部分主要是介绍TFlearn的学习

## 自定义一个简单的模型，具体代码见customize_model.py
### 核心部分:
1. 设计网络结构<br>
2. 得到prediction和loss<br>
3. 得到train_op<br>
4. 返回prediction,loss,train_op<br>
5. 将自己定义好的模型直接放入到learn.Estimator中，而且可以使用sklearn风格的fit和predict<br>
### 运行的效果：

## 使用TFlearn中的DNNClassifier， 具体代码见：DNNClassifier.py

### 运行效果：
{'loss': 0.6527705, 'accuracy': 0.6666667, 'labels/prediction_mean': 0.44106182, 'labels/actual_label_mean': 0.33333334, 'accuracy/baseline_label_mean': 0.33333334, 'auc': 0.9999995, 'auc_precision_recall': 0.999999, 'accuracy/threshold_0.500000_mean': 0.6666667, 'precision/positive_threshold_0.500000_mean': 0.0, 'recall/positive_threshold_0.500000_mean': 0.0, 'global_step': 10}


## 使用TFLearn中的Monitor模块， 具体代码见Monitor.py

