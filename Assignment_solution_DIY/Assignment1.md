<p align="center"><img src="tex/7b46180428e2456b52c2187d565f947b.svg?invert_in_darkmode&sanitize=true" align=middle width=318.91199505pt height=14.611878599999999pt/></p>  
  
**所有的代码题目对应的代码可查看对应文件夹AssignmentX_Code下的.py文件**
  
![1a](Assignment1-img/1a.jpg)  

**解：**
<p align="center"><img src="tex/35995193a6c6128f08c9bd37b6e628b8.svg?invert_in_darkmode&sanitize=true" align=middle width=490.57258634999994pt height=42.65424074999999pt/></p>  
即  
<p align="center"><img src="tex/62b524786a05e90fb15e7ea2c7959f36.svg?invert_in_darkmode&sanitize=true" align=middle width=222.53629034999997pt height=16.438356pt/></p>  
证毕  

![1b](Assignment1-img/1b.jpg)  

**解：**  
直接在代码中利用numpy实现即可。注意要先从<img src="tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>中减去每一行的最大值，这样在保证结果不变的情况下，所有的元素不大于0，不会出现`上溢出`，从而保证结果的正确性。具体可参考 http://www.hankcs.com/ml/computing-log-sum-exp.html  
  
![2a](Assignment1-img/2a.jpg)   

**解：**
<p align="center"><img src="tex/db9928d4fd9b64284c4f1a6d42ba3cc0.svg?invert_in_darkmode&sanitize=true" align=middle width=427.2343086pt height=39.59480249999999pt/></p>  

即<img src="tex/a1d10fd8a1c1198fcba88aa30b7107c3.svg?invert_in_darkmode&sanitize=true" align=middle width=58.41940499999999pt height=22.831056599999986pt/>函数的求导可以由其本身来表示。

![2b](Assignment1-img/2b.jpg)  

**解：**  
我们知道真实标记<img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>是one-hot向量，因此我们下面的推导都基于 <img src="tex/d188eb9ee9e3183afbdec8b468d271bf.svg?invert_in_darkmode&sanitize=true" align=middle width=46.28421599999999pt height=21.18721440000001pt/> ,且 <img src="tex/92fc33d17362605fd0aaa5e37d91e79f.svg?invert_in_darkmode&sanitize=true" align=middle width=87.63117659999999pt height=22.831056599999986pt/> ，即真实标记是 <img src="tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> .  

<p align="center"><img src="tex/88ef3df5488b0c66f5cf98311d9cf84d.svg?invert_in_darkmode&sanitize=true" align=middle width=206.7417165pt height=37.9216761pt/></p>  

其中：  
<p align="center"><img src="tex/cae0e852e541c146d64bfa168bd0ade4.svg?invert_in_darkmode&sanitize=true" align=middle width=209.23667655pt height=42.84447255pt/></p>  

接下来讨论 <img src="tex/d29eac8ed44d78c9fe9b67a26afabebc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.251585249999996pt height=30.648287999999997pt/> :  
1) <img src="tex/adb954c91e658590185031beeb908ca2.svg?invert_in_darkmode&sanitize=true" align=middle width=36.65622509999999pt height=22.831056599999986pt/>:  
<p align="center"><img src="tex/9b42f0140270a2e17adc2c751b84ff0f.svg?invert_in_darkmode&sanitize=true" align=middle width=250.53299205pt height=43.521805799999996pt/></p>  

则：  
<p align="center"><img src="tex/069026df3d257f09b250f873c8e2ecd8.svg?invert_in_darkmode&sanitize=true" align=middle width=337.05503534999997pt height=37.0084374pt/></p>

2) <img src="tex/4825b4d7040a1b0ab62a9ee3331b7287.svg?invert_in_darkmode&sanitize=true" align=middle width=36.65622509999999pt height=22.831056599999986pt/>:  
<p align="center"><img src="tex/48d03e3246e339d8124318b30d450715.svg?invert_in_darkmode&sanitize=true" align=middle width=213.55523969999996pt height=43.521805799999996pt/></p>  

则：  
<p align="center"><img src="tex/c011b1357fbd33e66ed3842137d01431.svg?invert_in_darkmode&sanitize=true" align=middle width=289.78262939999996pt height=37.0084374pt/></p>  

综上：  
<p align="center"><img src="tex/496235666ab69f2f691c6bac0134f782.svg?invert_in_darkmode&sanitize=true" align=middle width=205.1064147pt height=49.315569599999996pt/></p>  

或者：

<p align="center"><img src="tex/dcc4e9cbbbb091e8594bdd266757d597.svg?invert_in_darkmode&sanitize=true" align=middle width=134.31675345pt height=37.1910528pt/></p>  

![2c](Assignment1-img/2c.jpg)  

**解：**  
首先设：<img src="tex/7ae7b79130069ec6f6ab73744840a29a.svg?invert_in_darkmode&sanitize=true" align=middle width=106.50668654999998pt height=22.831056599999986pt/> 和 <img src="tex/cc62bf57a815668bdb239e4a0ed71b92.svg?invert_in_darkmode&sanitize=true" align=middle width=106.58281424999998pt height=22.831056599999986pt/>，那么前向传播的顺序依次为：  

<p align="center"><img src="tex/b4ed13c4a160555c336141297985afbe.svg?invert_in_darkmode&sanitize=true" align=middle width=106.50668655pt height=13.881256950000001pt/></p>
<p align="center"><img src="tex/287023cd062b28101901deca995ec2e0.svg?invert_in_darkmode&sanitize=true" align=middle width=121.18952339999998pt height=16.438356pt/></p>
<p align="center"><img src="tex/30fafd31a89230080c2fbe61fb7606fc.svg?invert_in_darkmode&sanitize=true" align=middle width=106.58281425pt height=13.881256950000001pt/></p>
<p align="center"><img src="tex/1121619c14f5e30af0b96bf02f2580bf.svg?invert_in_darkmode&sanitize=true" align=middle width=125.89248705pt height=16.438356pt/></p>
<p align="center"><img src="tex/02fe4132c041c5015951a02fdacd6a83.svg?invert_in_darkmode&sanitize=true" align=middle width=221.41387619999998pt height=36.6554298pt/></p>

现在求<img src="tex/3907d1fcf323c95464418c95ef0d1437.svg?invert_in_darkmode&sanitize=true" align=middle width=16.086209699999998pt height=28.92634470000001pt/>其实就是进行一次反向传播：  
<p align="center"><img src="tex/393336cd04583e9fb7f45894278fd735.svg?invert_in_darkmode&sanitize=true" align=middle width=128.08681545pt height=36.2778141pt/></p>
<p align="center"><img src="tex/c5bfaf6dbc0f5f48a25e535e752e154d.svg?invert_in_darkmode&sanitize=true" align=middle width=375.26687384999997pt height=36.2778141pt/></p>
<p align="center"><img src="tex/f6bad702a8f943db838dc1265181956f.svg?invert_in_darkmode&sanitize=true" align=middle width=375.75722579999996pt height=37.1910528pt/></p>
<p align="center"><img src="tex/91de91ed4fdc3d8790d5befe97938e51.svg?invert_in_darkmode&sanitize=true" align=middle width=435.43210094999995pt height=36.2778141pt/></p>

![2d](Assignment1-img/2d.jpg)  

**解：**  
(1) 从输入层到隐藏层，全连接共<img src="tex/0373fd6e72358d8b50f11e8b30d66f22.svg?invert_in_darkmode&sanitize=true" align=middle width=56.97704429999999pt height=22.465723500000017pt/>个，即<img src="tex/4c0c82cdc5d7bf2312fe6669d3f632f3.svg?invert_in_darkmode&sanitize=true" align=middle width=22.07767979999999pt height=22.465723500000017pt/>，加上<img src="tex/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>个偏置项，共<img src="tex/47df53dfe90e0bbe08fda80bcf0aa1f8.svg?invert_in_darkmode&sanitize=true" align=middle width=92.06820479999999pt height=22.465723500000017pt/>个。  
(2) 从隐藏层到输出层，共<img src="tex/b4f3dae50d17523f3305e04419370242.svg?invert_in_darkmode&sanitize=true" align=middle width=97.38265349999999pt height=22.465723500000017pt/>个。  
参数个数共：
<p align="center"><img src="tex/c3a6ccee4af0503fad9a9f335684b50b.svg?invert_in_darkmode&sanitize=true" align=middle width=235.93478864999997pt height=17.031940199999998pt/></p>

![2e](Assignment1-img/2e.jpg)  

![2f](Assignment1-img/2f.jpg)  

![2g](Assignment1-img/2g.jpg)  

![3a](Assignment1-img/3a.jpg)  

**解：**  
首先分析各个量的形状：<img src="tex/d62ec71e279fdd228438c1c2510a2071.svg?invert_in_darkmode&sanitize=true" align=middle width=204.16300575pt height=24.65753399999998pt/>，<img src="tex/9a0a8780188100cf9c4148b6d8f577b1.svg?invert_in_darkmode&sanitize=true" align=middle width=90.81408599999999pt height=22.831056599999986pt/>，其中<img src="tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/>为词典大小，<img src="tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/>为词向量的维度。   
我们设：  
<p align="center"><img src="tex/37dc1e765f2cd45a103d7260a9c18731.svg?invert_in_darkmode&sanitize=true" align=middle width=217.20188325pt height=78.93481695pt/></p>  

则：  
<p align="center"><img src="tex/39560c0f0230f197edc408f2fa17146a.svg?invert_in_darkmode&sanitize=true" align=middle width=337.36153935pt height=44.20028085pt/></p>  
<p align="center"><img src="tex/24dcaad0b833a7eace291aea01c4486b.svg?invert_in_darkmode&sanitize=true" align=middle width=115.47007395pt height=16.438356pt/></p>  
那么：  
<p align="center"><img src="tex/dfdbd84cb104f193267a60edcc777378.svg?invert_in_darkmode&sanitize=true" align=middle width=360.74444339999997pt height=36.2778141pt/></p>  

![3b](Assignment1-img/3b.jpg)  

**解：**  
可以先对<img src="tex/cacd1c2e0cf1e94e2d60dc2676941b37.svg?invert_in_darkmode&sanitize=true" align=middle width=22.549657349999993pt height=27.6567522pt/>求导：  
<p align="center"><img src="tex/d134f260efa7e0a936817f9bf0dd6f9e.svg?invert_in_darkmode&sanitize=true" align=middle width=377.91970425pt height=33.81208709999999pt/></p>  
  
那么对 <img src="tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/> 求导的结果对上式转置即可：  

<p align="center"><img src="tex/e952ff537506a2c23f4e2f290beddb2d.svg?invert_in_darkmode&sanitize=true" align=middle width=258.81103874999997pt height=33.81208709999999pt/></p>  
也可以表示为：  
<p align="center"><img src="tex/edfe303af652960976c525ae16d40084.svg?invert_in_darkmode&sanitize=true" align=middle width=204.65659004999998pt height=49.315569599999996pt/></p>  

![3c](Assignment1-img/3c.jpg)  

**解：**  
首先应该知道：  
<p align="center"><img src="tex/5e06979ca858ab24ee2031098991773f.svg?invert_in_darkmode&sanitize=true" align=middle width=175.9872312pt height=17.2895712pt/></p>  
<p align="center"><img src="tex/f15762519fda7bd2d93fda5532bd7f83.svg?invert_in_darkmode&sanitize=true" align=middle width=127.34007165pt height=16.438356pt/></p>
已知：  
<p align="center"><img src="tex/4e52cc783b4b4a5f9cbaeb6df74d77f5.svg?invert_in_darkmode&sanitize=true" align=middle width=351.19346459999997pt height=48.18280005pt/></p>  

直接求导即可：  
<p align="center"><img src="tex/8d04c0e642b7d21b4d52f5923d6ea079.svg?invert_in_darkmode&sanitize=true" align=middle width=579.4024367999999pt height=48.18280005pt/></p>  
<p align="center"><img src="tex/4de1b8fb9da884d11844a055e83a33ba.svg?invert_in_darkmode&sanitize=true" align=middle width=232.62952185pt height=49.315569599999996pt/></p>  

![3d](Assignment1-img/3d.jpg)

**解：**  
根据题目的提示可知，我们可以设<img src="tex/996179619ce79a5abf30eb63a96041ec.svg?invert_in_darkmode&sanitize=true" align=middle width=55.577925149999984pt height=24.65753399999998pt/>为损失函数，等价于前面的<img src="tex/deb8f8f12ea1a0b165cec5398720a2d0.svg?invert_in_darkmode&sanitize=true" align=middle width=91.51519739999999pt height=22.465723500000017pt/>或者<img src="tex/be844990d2cd4286a8b6a64683f55ea2.svg?invert_in_darkmode&sanitize=true" align=middle width=82.81501469999999pt height=22.465723500000017pt/>，而<img src="tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/>对变量的求导我们前面已经做过，所以这里直接使用<img src="tex/fdb7b89ce793e3d96f78ba99ea28db35.svg?invert_in_darkmode&sanitize=true" align=middle width=51.2358pt height=33.20539859999999pt/>代替即可，不用再进一步求导展开。  
(1) **skip-gram模型**  
<p align="center"><img src="tex/b5bf85ca28230626517e7ede6414255f.svg?invert_in_darkmode&sanitize=true" align=middle width=384.49203374999996pt height=39.26959575pt/></p>
<p align="center"><img src="tex/3d026c07194fdebfc47d71871850cf80.svg?invert_in_darkmode&sanitize=true" align=middle width=230.07871589999996pt height=45.4586385pt/></p>  
<p align="center"><img src="tex/f7a4595439418bc24102ba2364327e0b.svg?invert_in_darkmode&sanitize=true" align=middle width=231.72739259999997pt height=45.4586385pt/></p>   
<p align="center"><img src="tex/ecd2a47dbb0854068e70ebfbdc29bb07.svg?invert_in_darkmode&sanitize=true" align=middle width=100.69202055pt height=38.5152603pt/></p>   
   
\(2) **CBOW模型**  
因为CBOW模型是根据多个背景词预测一个中心词，又因为<img src="tex/a05e71f7cbcf6e5341b31140bc3b1ada.svg?invert_in_darkmode&sanitize=true" align=middle width=25.63935659999999pt height=24.65753399999998pt/>惩罚函数是形如`(一个词，一个词)`的形式，所以要把多个背景词变成一个词，那么一种有效的方式就是把这些背景词的词向量求平均便得到了一个词向量。  
<p align="center"><img src="tex/30531f7a285d218ab27cb03bc819d107.svg?invert_in_darkmode&sanitize=true" align=middle width=150.16239149999998pt height=39.26959575pt/></p>
<p align="center"><img src="tex/084c65e05bf6a91c7160572e84fa25cb.svg?invert_in_darkmode&sanitize=true" align=middle width=250.50275745pt height=16.438356pt/></p>  
那么：  
<p align="center"><img src="tex/7cb71a151514889cea95584477d08a0e.svg?invert_in_darkmode&sanitize=true" align=middle width=118.12771244999999pt height=34.7253258pt/></p>  
<p align="center"><img src="tex/963d9f11a93720af38de98c8b4cdcc00.svg?invert_in_darkmode&sanitize=true" align=middle width=308.9767197pt height=36.2778141pt/></p>  
<p align="center"><img src="tex/92c22062fa543b7222e3880b386cab16.svg?invert_in_darkmode&sanitize=true" align=middle width=510.36440235pt height=39.428498999999995pt/></p>  

![3e](Assignment1-img/3e.jpg)  

![3f](Assignment1-img/3f.jpg)  

![3g](Assignment1-img/3g.jpg)  
**解：**   
我本地共训练了**5**+个小时。  
输出的结果为：  
![3g_ans](Assignment1-img/q3_word_vectors.png)
   

![3h](Assignment1-img/3h.jpg)  

![4a](Assignment1-img/4a.jpg)  
**解：**
按题目要求实现即可。

![4b](Assignment1-img/4b.jpg)  
**解：** 引入正则化可以降低模型复杂度，进而避免过拟合，以提升泛化能力。  

![4c](Assignment1-img/4c.jpg)  
**解：** 注意是按照模型的验证集准确率来选择最优模型。  

![4d](Assignment1-img/4d.jpg)  
**解：** 我的本地答案：  
(1) 使用自己训练的词向量的结果
```
Best regularization value: 7.05E-04
Test accuracy (%): 30.361991
dev accuracy (%): 32.698
```
  
(2) 使用预训练的词向量的结果
```
Best regularization value: 1.23E+01
Test accuracy (%): 37.556561
dev accuracy (%): 37148
```
使用预训练的词向量的效果更好的原因：
* 其数据量大。
* 训练充分。
* 其采用的为GloVe,该模型利用全局的信息。
* 维度高。

![4e](Assignment1-img/4e.jpg)  
**解：**  
![4e_ans](Assignment1-img/q4_reg_v_acc.png)   
解释：随着正则化因子的增大，最终所得的模型越简单，拟合能力差，出现欠拟合，导致两者的准确率下降。  

![4f](Assignment1-img/4f.jpg)  
**解：**  
![4f_ans](Assignment1-img/q4_dev_conf.png)  
  

![4g](Assignment1-img/4g.jpg)  