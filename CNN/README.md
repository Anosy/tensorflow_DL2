# 本部分主要是介绍卷积神经网络的构建

## 第一部分: 简单的卷积神经网络, 具体代码见：sample_CNN.py
### 网络框架：
1.卷积层：卷积核的大小：5*5*32，步长为1*1，使用padding='SAME'，得到的最后输出大小为28*28*32<br>
2.池化层：池化窗口的大小为： 2*2， 步长为 2*2， 使用padding='SAME'，得到的最后输出大小为14*14*32<br>
3.卷积层：卷积核的大小：5*5*64，步长为1*1，使用padding='SAME'，得到的最后输出大小为14*14*64<br>
4.池化层：池化窗口的大小为： 2*2， 步长为 2*2， 使用padding='SAME'，得到的最后输出大小为7*7*32<br>
5.全连接层<br>
6.全连接层<br>
### 结果：
epoch  100, training accuracy=  0.96<br>
epoch  100, training loss=  0.0954716<br>
epoch  200, training accuracy=  0.98<br>
epoch  200, training loss=  0.0638389<br>
epoch  300, training accuracy=  1<br>
epoch  300, training loss=  0.0559012<br>
epoch  400, training accuracy=  0.94<br>
......<br>
epoch  1700, training accuracy=  0.98<br>
epoch  1700, training loss=  0.111496<br>
epoch  1800, training accuracy=  1<br>
epoch  1800, training loss=  0.00470316<br>
epoch  1900, training accuracy=  0.96<br>
epoch  1900, training loss=  0.0594318<br>
epoch  2000, training accuracy=  0.98<br>
epoch  2000, training loss=  0.0544735<br>
### 结果图：
**Graph**<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/sample_graph.png)<br>
**Accuracy**<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/simple_accuracy.png)<br>
**Loss**<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/sample_loss.png)<br>
**Image**<br>
Input：<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/simple_input.png)<br>
Conv1:<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/simple_conv1.png)<br>
Relu1:<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/simple_relu1.png)<br>
Conv2:<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/simple_conv2.png)<br>
Relu2:<br>
![](https://github.com/Anosy/tensorflow_DL2/blob/master/CNN/result_picture/simple_relu2.png)<br>

## 第二部分：使用CNN来为CIFAR-10数据来做分类
### 网络框架：
1.卷积层，卷积核的大小为5*5<br>
2.池化层，窗口的大小为3*3， 步长为2*2<br>
3.LRN层局部响应归一化<br>
4.卷积层，卷积核的大小为5*5<br>
5.LRN层局部响应归一化<br>
6.池化层，窗口的大小3*3，步长为2*2<br>
7.全连接层，隐含层节点384<br>
8.全连接层，隐含层节点192<br>
9.softmax层，输出节点10<br>
### 核心代码以及结构：

        # 加载数据
        cifar10.maybe_download_and_extract() 
        # 网络结构
        ......
        # 计算损失函数，添加上正则化L2
        ......
        # 优化训练过程
        ......
        # 求输出结果top1的正确率
        # 注：这里logits输入的是one-hot编码形式的，label_in为单一数值形式的。如果为one-hot形式，需要使用tf.argmax(label_in, 1)
        top_k_op = tf.nn.in_top_k(logits, label_in, 1) 
        # 启动线程，在图像数据增强队列例使用了16个线程进行加速。
        tf.train.start_queue_runners() 
        # 训练过程
        ......
        # 测试过程
        ......
        
### 运行结果:
step 0, loss=4.68(3.6 exaples/sec; 35.456 sec/batch)<br>
step 10, loss=3.74(1365.7 exaples/sec; 0.094 sec/batch)<br>
step 20, loss=3.21(1365.6 exaples/sec; 0.094 sec/batch)<br>
step 30, loss=2.71(1365.6 exaples/sec; 0.094 sec/batch)<br>
step 40, loss=2.52(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 50, loss=2.32(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 60, loss=2.11(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 70, loss=2.05(1365.7 exaples/sec; 0.094 sec/batch)<br>
step 80, loss=2.11(1365.7 exaples/sec; 0.094 sec/batch)<br>
step 90, loss=1.95(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 100, loss=2.02(1614.2 exaples/sec; 0.079 sec/batch)<br>
......<br>
step 4900, loss=0.93(1307.9 exaples/sec; 0.098 sec/batch)<br>
step 4910, loss=0.83(1365.6 exaples/sec; 0.094 sec/batch)<br>
step 4920, loss=0.93(1365.6 exaples/sec; 0.094 sec/batch)<br>
step 4930, loss=0.95(1638.6 exaples/sec; 0.078 sec/batch)<br>
step 4940, loss=0.95(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 4950, loss=1.07(819.4 exaples/sec; 0.156 sec/batch)<br>
step 4960, loss=0.92(1365.7 exaples/sec; 0.094 sec/batch)<br>
step 4970, loss=0.86(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 4980, loss=0.99(1638.8 exaples/sec; 0.078 sec/batch)<br>
step 4990, loss=1.17(1365.7 exaples/sec; 0.094 sec/batch)<br>
precision @ 1 = 0.756<br>
**可以从结果中看出在GPU1050的情况下，每秒能训练1300个样本左右，而且基本上一个batch的样本只要0.9秒左右，最后模型的精度为top-1:75.6%**


