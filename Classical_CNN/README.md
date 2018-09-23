# 本部分主要介绍的几种经典的卷积神经网络
## 实验环境：GeForce GTX 1080
## AlexNET
### 网络结构
![Alex Net](https://github.com/Anosy/tensorflow_DL2/blob/master/Classical_CNN/result_picture/AlexNet.jpg)
### 运行结果：
**每层输出的tensor的shape**<br>
Variable  [32, 224, 224, 3]<br>
conv1  [32, 56, 56, 64]<br>
pool1  [32, 27, 27, 64]<br>
conv2  [32, 27, 27, 192]<br>
pool2  [32, 13, 13, 192]<br>
conv3  [32, 13, 13, 384]<br>
conv4  [32, 13, 13, 256]<br>
conv45  [32, 13, 13, 256]<br>
pool3  [32, 6, 6, 256]<br>
fc1  [32, 4096]<br>
fc2  [32, 4096]<br>
softmax  [32, 1000]<br>
<br>
**运行的时间**<br>
2018-09-22 09:55:29.859338: step 0, duration = 0.016<br>
2018-09-22 09:55:29.953067: step 10, duration = 0.016<br>
2018-09-22 09:55:30.046758: step 20, duration = 0.016<br>
2018-09-22 09:55:30.140487: step 30, duration = 0.016<br>
2018-09-22 09:55:30.218635: step 40, duration = 0.000<br>
2018-09-22 09:55:30.312360: step 50, duration = 0.000<br>
2018-09-22 09:55:30.406049: step 60, duration = 0.000<br>
2018-09-22 09:55:30.499813: step 70, duration = 0.000<br>
2018-09-22 09:55:30.593541: step 80, duration = 0.000<br>
2018-09-22 09:55:30.687270: step 90, duration = 0.000<br>
2018-09-22 09:55:30.780963: Forward across 100 steps, 0.009 +/- 0.01 sec /batch<br>
2018-09-22 09:55:31.530786: step 0, duration = 0.031<br>
2018-09-22 09:55:31.858834: step 10, duration = 0.031<br>
2018-09-22 09:55:32.186882: step 20, duration = 0.031<br>
2018-09-22 09:55:32.514952: step 30, duration = 0.031<br>
2018-09-22 09:55:32.843017: step 40, duration = 0.031<br>
2018-09-22 09:55:33.171027: step 50, duration = 0.031<br>
2018-09-22 09:55:33.499110: step 60, duration = 0.031<br>
2018-09-22 09:55:33.827158: step 70, duration = 0.031<br>
2018-09-22 09:55:34.155206: step 80, duration = 0.031<br>
2018-09-22 09:55:34.483254: step 90, duration = 0.031<br>
2018-09-22 09:55:34.764424: Forward-backward across 100 steps, 0.033 +/- 0.00 sec /batch<br>

## VGGNet-16
### 网络结构
![VGGNet-16](https://github.com/Anosy/tensorflow_DL2/blob/master/Classical_CNN/result_picture/VGG-16.png)
### 运行结果：
**每层输出的tensor的shape**<br>
poo11  [32, 112, 112, 64]<br>
pool2  [32, 56, 56, 128]<br>
pool3  [32, 28, 28, 256]<br>
pool4  [32, 14, 14, 512]<br>
pool5  [32, 7, 7, 512]<br>
resh1  [32, 25088]<br>
fc6_drop/mul  [32, 4096]<br>
fc7_drop/mul  [32, 4096]<br>
fc8  [32, 1000]<br>
softmax  [32, 1000]<br>
<br>
**运行的时间**<br>
2018-09-22 11:17:54.601494: step 0, duration = 0.141<br>
2018-09-22 11:17:55.991794: step 10, duration = 0.141<br>
2018-09-22 11:17:57.382127: step 20, duration = 0.141<br>
2018-09-22 11:17:58.772389: step 30, duration = 0.141<br>
2018-09-22 11:18:00.147098: step 40, duration = 0.141<br>
2018-09-22 11:18:01.537364: step 50, duration = 0.141<br>
2018-09-22 11:18:02.927694: step 60, duration = 0.141<br>
2018-09-22 11:18:04.317992: step 70, duration = 0.141<br>
2018-09-22 11:18:05.708277: step 80, duration = 0.141<br>
2018-09-22 11:18:07.098576: step 90, duration = 0.141<br>
2018-09-22 11:18:08.348295: Forward across 100 steps, 0.139 +/- 0.01 sec /batch<br>
2018-09-22 11:18:14.909219: step 0, duration = 0.453<br>
2018-09-22 11:18:19.314431: step 10, duration = 0.453<br>
2018-09-22 11:18:23.735266: step 20, duration = 0.437<br>
2018-09-22 11:18:28.202968: step 30, duration = 0.437<br>
2018-09-22 11:18:32.623834: step 40, duration = 0.453<br>
2018-09-22 11:18:37.044635: step 50, duration = 0.437<br>
2018-09-22 11:18:41.527959: step 60, duration = 0.484<br>
2018-09-22 11:18:45.964432: step 70, duration = 0.453<br>
2018-09-22 11:18:50.400903: step 80, duration = 0.437<br>
2018-09-22 11:18:54.852943: step 90, duration = 0.437<br>
2018-09-22 11:18:58.852024: Forward-backward across 100 steps, 0.444 +/- 0.01 sec /batch<br>

## Inception-V3
### 网络结构
![Inception-V3](https://github.com/Anosy/tensorflow_DL2/blob/master/Classical_CNN/result_picture/inception-v3.png)
### 运行结果：
**每层输出的tensor的shape**<br>
InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool  [32, 35, 35, 192]<br>
InceptionV3/InceptionV3/Mixed_5b/concat  [32, 35, 35, 256]<br>
InceptionV3/InceptionV3/Mixed_5c/concat  [32, 35, 35, 288]<br>
InceptionV3/InceptionV3/Mixed_5d/concat  [32, 35, 35, 288]<br>
InceptionV3/InceptionV3/Mixed_6a/concat  [32, 17, 17, 768]<br>
InceptionV3/InceptionV3/Mixed_6b/concat  [32, 17, 17, 768]<br>
InceptionV3/InceptionV3/Mixed_6c/concat  [32, 17, 17, 768]<br>
InceptionV3/InceptionV3/Mixed_6d/concat  [32, 17, 17, 768]<br>
InceptionV3/InceptionV3/Mixed_6e/concat  [32, 17, 17, 768]<br>
InceptionV3/InceptionV3/Minxed_7a/concat  [32, 8, 8, 1280]<br>
InceptionV3/InceptionV3/Mixed_7b/concat  [32, 8, 8, 2048]<br>
InceptionV3/InceptionV3/Mixed_7c/concat  [32, 8, 8, 2048]<br>
InceptionV3/Logits/Dropout_1b/Identity  [32, 1, 1, 2048]<br>
InceptionV3/Logits/Conv2d_1c_1x1/BiasAdd  [32, 1, 1, 1000]<br>
InceptionV3/Logits/SpatialSqueeze  [32, 1000]<br>
<br>
**运行的时间**<br>
2018-09-23 14:39:45.069105: step 0, duration = 0.154<br>
2018-09-23 14:39:46.605006: step 10, duration = 0.154<br>
2018-09-23 14:39:48.143904: step 20, duration = 0.156<br>
2018-09-23 14:39:49.688786: step 30, duration = 0.153<br>
2018-09-23 14:39:51.224665: step 40, duration = 0.154<br>
2018-09-23 14:39:52.762529: step 50, duration = 0.152<br>
2018-09-23 14:39:54.311409: step 60, duration = 0.155<br>
2018-09-23 14:39:55.848276: step 70, duration = 0.155<br>
2018-09-23 14:39:57.384198: step 80, duration = 0.156<br>
2018-09-23 14:39:58.936026: step 90, duration = 0.154<br>
2018-09-23 14:40:00.320316: Forward across 100 steps, 0.154 +/- 0.00 sec /batch<br>

## RestNet-V2
### 网络结构
![RestNet-V2](https://github.com/Anosy/tensorflow_DL2/blob/master/Classical_CNN/result_picture/RestNet-V2.png)
### 运行结果：
**运行的时间**<br>
2018-09-23 15:44:47.156761: step 0, duration = 0.236<br>
2018-09-23 15:44:49.481545: step 10, duration = 0.234<br>
2018-09-23 15:44:51.834289: step 20, duration = 0.238<br>
2018-09-23 15:44:54.163024: step 30, duration = 0.231<br>
2018-09-23 15:44:56.490814: step 40, duration = 0.238<br>
2018-09-23 15:44:58.823560: step 50, duration = 0.230<br>
2018-09-23 15:45:01.171282: step 60, duration = 0.234<br>
2018-09-23 15:45:03.515014: step 70, duration = 0.233<br>
2018-09-23 15:45:05.853758: step 80, duration = 0.237<br>
2018-09-23 15:45:08.204472: step 90, duration = 0.235<br>
2018-09-23 15:45:10.303858: Forward across 100 steps, 0.234 +/- 0.003 sec / batch<br>
