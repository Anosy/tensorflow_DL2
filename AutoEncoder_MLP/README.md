# 本部分主要是介绍自编码器以及MLP
## 第一部分：自编码器
### 原理：
  在神经网络中，随着神经网络的深入，越到后面的层，其提取的信息就越是高阶的信息。所以，如果我希望提取到更加高级的信息就要可以使用更加深层的网络结构。如果有很多的标注数据的情况下，可以使用神经网络，但是如果没有很多的标注数据怎么办？这时候就可以考虑使用自编码器。<br>
  自编码器，顾名思义就是可以使用自己的高阶特征来编码自己。<br>
  ![自编码原理图](https://github.com/Anosy/tensorflow_DL2/tree/master/AutoEncoder_MLP/result_picture/AutoEncoder_principle.jpg)<br
  从上图可以看出，自编码器就是由编码器和解码器构成的，而且输入和输出是一样的。<br>
  目前自编码器的应用主要有两个方面，第一是数据去噪，第二是为进行可视化而降维。配合适当的维度和稀疏约束，自编码器可以学习到比PCA等技术更有意思的数据投影。<br>

### 核心代码：
 
		# xavier_init 初始化方法，这种方法的作用就是将参数初始化到不大不小的程度，既不会导致梯度消失也不会导致梯度爆炸
		# Xavier就是要让权重满足mean=0, stdevv=2/(n_in + n_out), 这里使用均有分布到等效实现
		def xavier_init(fan_in, fan_out, constant = 1):
			low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
			high = constant * np.sqrt(6.0 / (fan_in + fan_out))
			return tf.random_uniform ((fan_in, fan_out),
				minval = low, maxval = high, dtype = tf.float32)
				
### 代码运行结果：
Epoch: 0001 cost= 154.314405948<br>
Epoch: 0002 cost= 93.146246831<br>
Epoch: 0003 cost= 83.318605890<br>
Epoch: 0004 cost= 79.824505313<br>
Epoch: 0005 cost= 78.432383230<br>
Epoch: 0006 cost= 81.154963099<br>
Epoch: 0007 cost= 76.075486981<br>
Epoch: 0008 cost= 71.695870428<br>
Epoch: 0009 cost= 64.134568080<br>
Epoch: 0010 cost= 66.533526012<br>
Epoch: 0011 cost= 67.032167334<br>
Epoch: 0012 cost= 68.303277979<br>
Epoch: 0013 cost= 60.101727362<br>
Epoch: 0014 cost= 64.757767978<br>
Epoch: 0015 cost= 66.277053089<br>
Epoch: 0016 cost= 62.449909806<br>
Epoch: 0017 cost= 66.674848384<br>
Epoch: 0018 cost= 65.172000919<br>
Epoch: 0019 cost= 58.674013534<br>
Epoch: 0020 cost= 63.972098917<br>
Total coat:181327.0<br>

## 第二部分：多层感知机 
### 代码运行结果：
epoch 0, loss = 2.790883779525757<br>
epoch 100, loss = 0.38568365573883057<br>
epoch 200, loss = 0.11886562407016754<br>
epoch 300, loss = 0.21469536423683167<br>
epoch 400, loss = 0.19315999746322632<br>
epoch 500, loss = 0.10259602218866348<br>
......<br>
epoch 2500, loss = 0.05385969206690788<br>
epoch 2600, loss = 0.06590429693460464<br>
epoch 2700, loss = 0.04444016143679619<br>
epoch 2800, loss = 0.020722856745123863<br>
epoch 2900, loss = 0.014302815310657024<br>
epoch 3000, test accuracy = 0.9786999821662903<br>
**Graph**<br>
![](https://github.com/Anosy/tensorflow_DL2/tree/master/AutoEncoder_MLP/result_picture/two_layer_mnist.png)<br>
**Loss**<br>
![](https://github.com/Anosy/tensorflow_DL2/tree/master/AutoEncoder_MLP/result_picture/accuracy.png)<br>
**Acc**<br>
![](https://github.com/Anosy/tensorflow_DL2/tree/master/AutoEncoder_MLP/result_picture/cross_entropy.png)<br>
**weight**<br>
![](https://github.com/Anosy/tensorflow_DL2/tree/master/AutoEncoder_MLP/result_picture/weight.png)<br>




