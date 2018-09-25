# 本部分主要介绍的是Word2Vec的实现和LSTM

## W2V，具体代码见w2v.py

### 代码结构：
1. 下载文件，将文件保存为text8.zip<br>
2. 读取文件内容，获取所有词汇<br>
3. 创建词汇表<br>
4. 生成batch<br>
......<br>

### 实验结果：
初始迭代：<br>
average loss at step: 0 : 290.306884765625<br>
(16, 50000)<br>
(16, 50000)<br>
nearest to states: astounding, kpa, threatened, inhabits, factum, deception, appeasement, aggressively,<br>
nearest to united: meet, emerging, inhospitable, crater, busting, armory, mu, gilda,<br>
nearest to see: forms, termini, taft, projectors, current, fibs, leiserson, jamie,<br>
nearest to who: dietary, wheatstone, waveforms, slovenian, variably, kaye, undated, ifr,<br>
nearest to however: devastating, un, mifune, acta, ncipe, ely, bua, teamed,<br>
nearest to on: hypertalk, heterotic, chaos, practitioner, sagas, crackdown, willamette, drift,<br>
nearest to has: answer, rhotic, flowers, copyrighted, mutual, physically, bravo, olsen,<br>
nearest to war: sample, functionals, arranging, splitflag, seismic, kiev, sfsr, required,<br>
nearest to over: traversed, activating, parishes, cubewanos, traynor, tonnage, commanders, sensei,<br>
nearest to than: ilium, implanted, king, honeys, penicillium, jamil, easton, authorities,<br>
nearest to world: kornbluth, illustrator, debate, annunzio, balboa, cavity, gimp, biplanes,<br>
nearest to four: reportage, constriction, previously, representationalism, rewarding, kernow, interstellar, foothold,<br>
nearest to five: streams, renamed, tunny, enrique, rivets, sherman, fools, fugitives,<br>
nearest to had: pumps, gnutella, danish, retraction, hypoglycemic, uptime, computations, axial,<br>
nearest to american: karabakh, dimensionless, bullion, hedonistic, ninth, midwives, payable, mchenry,<br>
nearest to seven: wing, andromeda, dreadnought, crater, ascertain, warrior, protracted, eureka,<br>
......<br>
最终迭代：<br>
average loss at step: 92000 : 4.70764683830738<br>
average loss at step: 94000 : 4.635076761841774<br>
average loss at step: 96000 : 4.714563299655914<br>
average loss at step: 98000 : 4.626688945412636<br>
average loss at step: 100000 : 4.663026185274124<br>
(16, 50000)<br>
(16, 50000)<br>
nearest to states: races, inhabits, kpa, cegep, astounding, astronauts, deception, dismiss,<br>
nearest to united: emerging, inhospitable, through, regeneration, including, dmd, callithrix, reintroducing,<br>
nearest to see: amalthea, epistemological, mitral, callithrix, pulau, six, although, cegep,<br>
nearest to who: he, they, we, there, which, and, plagiarism, also,<br>
nearest to however: but, although, that, thibetanus, when, and, which, mitral,<br>
nearest to on: in, at, through, upon, mitral, during, microcebus, against,<br>
nearest to has: had, have, was, is, since, although, abstract, having,<br>
nearest to war: cpr, usa, bandai, required, kiev, seismic, arranging, antiparticles,<br>
nearest to over: mitral, thibetanus, armouries, tonnage, three, condita, abitibi, dancing,<br>
nearest to than: or, and, mpd, bastiat, much, weighty, but, coleman,<br>
nearest to world: kornbluth, callithrix, charlestown, illustrator, xx, misses, neutrality, thaler,<br>
nearest to four: five, six, seven, three, eight, two, nine, zero,<br>
nearest to five: four, seven, six, eight, three, zero, nine, two,<br>
nearest to had: has, have, was, were, since, callithrix, when, been,<br>
nearest to american: british, french, canadian, microcebus, pulau, tamias, bayreuth, layman,<br>
nearest to seven: eight, six, five, four, nine, three, zero, thibetanus,<br>

### 结果分析：
从最终的结果可以看出，在迭代过程后得到的结果不错！<br>
![tsne](https://github.com/Anosy/tensorflow_DL2/blob/master/Word2vec_LSTM/tsne.png)<br>

## LSTM, 具体代码见：tensorflow_Bidirectional_LSTM_Classifier.py

### 实验结果：
iter1280,loss=2.091945,accuracy=0.34375<br>
iter2560,loss=1.178941,accuracy=0.59375<br>
iter3840,loss=0.878220,accuracy=0.71875<br>
iter5120,loss=0.616769,accuracy=0.80469<br>
iter6400,loss=0.516001,accuracy=0.82031<br>
......<br>
iter394240,loss=0.040718,accuracy=0.99219<br>
iter395520,loss=0.006640,accuracy=1.00000<br>
iter396800,loss=0.050201,accuracy=0.99219<br>
iter398080,loss=0.005117,accuracy=1.00000<br>
iter399360,loss=0.063068,accuracy=0.99219<br>


