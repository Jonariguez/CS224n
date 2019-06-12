<p align="center"><img src="/tex/3a9cc8bc8faeda1b348eb3e29d0912fb.svg?invert_in_darkmode&sanitize=true" align=middle width=319.82533935pt height=14.611878599999999pt/></p>  

所有的代码题目对应的代码可查看对应文件夹Assignment2_Code下的.py文件  

![1a](Assignment2-img/1a.jpg)

**解:**  
（提示使用keepdims参数会方便一些哦。  ）

![1b](Assignment2-img/1b.jpg)

**解:**  
（积累知识：  
* tf.multiply()为元素级的乘法，要求形状相同。  
* tf.matmul()为矩阵乘法。  
两者都要求两个矩阵的元素类型必须相同。    

） 

![1c](Assignment2-img/1c.jpg)  

**解:**  
占位符(placeholder)和feed_dict可以在运行时动态地向计算图“喂”数据。（TensorFlow采用的是静态图）  

![1d](Assignment2-img/1d.jpg)  

![1e](Assignment2-img/1e.jpg)  

**解:**  
TensorFlow的自动梯度，是指我们使用时只需要定义图的节点就好了，不用自己实现求解梯度，反向传播和求导由TensorFlow自动完成。  
  

![2](Assignment2-img/2.jpg)  

![2a](Assignment2-img/2a.jpg)  
**解：**  

|stack|buffer|new dependency|transition|
|-----|------|--------------|----------|
|[ROOT,parsed,this]|[sentence,correctly]| |SHIFT|
|[ROOT,parsed,this,sentence]|[correctly]| |SHIFT|
|[ROOT,parsed,sentence]|[correctly]|sentence -> this |LEFT-ARC|
|[ROOT,parsed]|[correctly]|parsed -> sentence |RIGHT-ARC|
|[ROOT,parsed,correctly]|[]| |SHIFT|
|[ROOT,parsed]|[]|parsed -> correctly |RIGHT-ARC|
|[ROOT]|[]|ROOT -> parsed |RIGHT-ARC|  
  

![2b](Assignment2-img/2b.jpg)  
**解：**  
共2n步  
* 每个词都要进入stack中，故要有n步SHIFT操作。  
* 最终stack中只剩ROOT，即每一次ARC会从stack中删掉一个词，故共有n步LEFT-ARC和RIGHT-ARC操作。  

   
![2c](Assignment2-img/2c.jpg)  

![2d](Assignment2-img/2d.jpg)  

![21](Assignment2-img/21.jpg)  

![2e](Assignment2-img/2e.jpg)

![2f](Assignment2-img/2f.jpg)
**解：**
<p align="center"><img src="/tex/46e2b7aae3a5afdebf766b596d22f986.svg?invert_in_darkmode&sanitize=true" align=middle width=460.8633397499999pt height=21.0684936pt/></p>  
即推导出：  
<p align="center"><img src="/tex/9c20fb5007bbf584389ad7f53fd407e3.svg?invert_in_darkmode&sanitize=true" align=middle width=97.2824622pt height=37.6933392pt/></p>  

![2f](Assignment2-img/2f.jpg)  

![2g1](Assignment2-img/2g1.jpg)  

**解:**  
因为其实<img src="/tex/273457f251a6f8920e7b6c485c28b74f.svg?invert_in_darkmode&sanitize=true" align=middle width=15.75334034999999pt height=14.611878600000017pt/>是之前全部梯度(更新量)的加权平均，更能体现梯度的整体变化。因为这样减小了更新量的方差，避免了梯度振荡。  
<img src="/tex/2a8ee4ae67ff6ec439274b277917dea8.svg?invert_in_darkmode&sanitize=true" align=middle width=16.74544574999999pt height=22.831056599999986pt/>一般要接近1。  

![2g2](Assignment2-img/2g2.jpg)  
**解：**  
* 更新量<img src="/tex/273457f251a6f8920e7b6c485c28b74f.svg?invert_in_darkmode&sanitize=true" align=middle width=15.75334034999999pt height=14.611878600000017pt/>: 对梯度(更新量)进行滑动平均  
* 学习率<img src="/tex/f6fc3ac36dff143d4aac9d145fadc77e.svg?invert_in_darkmode&sanitize=true" align=middle width=10.239687149999991pt height=14.611878600000017pt/>: 对梯度的平方进行滑动平均  

梯度平均最小的参数的更新量最大，也就是说，在损失函数相对于它们的梯度很小的时候也能快速收敛。即在平缓的地方也能快递移动到最优解。  

![2h](Assignment2-img/2h.jpg)  
**解：**  
我的结果为  
```
Epoch 10 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0654
Evaluating on dev set - dev UAS: 88.37
New best dev UAS! Saving model in ./data/weights/parser.weights

===========================================================================
TESTING
===========================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set
- test UAS: 88.84
```
运行时间：15分钟。  


![3](Assignment2-img/3.jpg)
**题目解读**  
先明确题目中各个量的维度：  
由题目可知，<img src="/tex/a206b03529b8e2e1d83c536f1613a930.svg?invert_in_darkmode&sanitize=true" align=middle width=24.63481184999999pt height=29.190975000000005pt/>是one-hot**行向量**，且**隐藏层也是行向量的形式**。  
则可得：  
<p align="center"><img src="/tex/907e89573151afff7ef3174c8f54a7d8.svg?invert_in_darkmode&sanitize=true" align=middle width=92.64852629999999pt height=16.06010835pt/></p>  
<p align="center"><img src="/tex/9fff7951f496d0ac3f951f954dcba309.svg?invert_in_darkmode&sanitize=true" align=middle width=91.87277385pt height=16.06010835pt/></p>  

<img src="/tex/8d45f165fcfa220512212b23cb968666.svg?invert_in_darkmode&sanitize=true" align=middle width=23.889029999999988pt height=29.190975000000005pt/>是输出，即每个单词的概率分布(softmax之后)，那么：  
<p align="center"><img src="/tex/f019d12e87955967d66fb71b61c98b66.svg?invert_in_darkmode&sanitize=true" align=middle width=91.90273785pt height=18.6137556pt/></p>  
然后我们就可以得到：  
<p align="center"><img src="/tex/3d702000558215e07db7749d405c5cf4.svg?invert_in_darkmode&sanitize=true" align=middle width=78.66960089999999pt height=16.06010835pt/></p>  
<p align="center"><img src="/tex/1407a282fb840e8c35e8f42a730f10cc.svg?invert_in_darkmode&sanitize=true" align=middle width=79.34881185pt height=16.06010835pt/></p>  
<p align="center"><img src="/tex/fa9d9622c8781ec0ca11db2b5605a90d.svg?invert_in_darkmode&sanitize=true" align=middle width=75.146445pt height=15.420842249999998pt/></p>  
<p align="center"><img src="/tex/06efe749984ddc95f6aa31b05ee57a07.svg?invert_in_darkmode&sanitize=true" align=middle width=93.15934979999999pt height=15.29299695pt/></p>  
<p align="center"><img src="/tex/71a399d3763feb42be65b0330611b557.svg?invert_in_darkmode&sanitize=true" align=middle width=80.76920445pt height=17.11602255pt/></p>  
<p align="center"><img src="/tex/9b82d47b2e9f7b5f67902ff272d32f8f.svg?invert_in_darkmode&sanitize=true" align=middle width=92.02720065pt height=16.06010835pt/></p>  
<p align="center"><img src="/tex/28ffdffc13e09a8fcdd517d0d5b943f7.svg?invert_in_darkmode&sanitize=true" align=middle width=81.62108459999999pt height=17.883133949999998pt/></p>  

其中<img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/>是词向量的长度，也就是代码中的<img src="/tex/cf9312674c7bfc60947212df5bd20e21.svg?invert_in_darkmode&sanitize=true" align=middle width=80.66041664999999pt height=22.831056599999986pt/>。  
在清楚了上面各矩阵的维度之后的求导才会更清晰。  

因为句子的长度不一，然后损失函数是针对一个单词所计算的，然后求和之后是对整个句子的损失，故要对损失函数求平均以得到每个单词的平均损失才行。  

![3a](Assignment2-img/3a.jpg)  
**解：**  
由于标签<img src="/tex/6a36c32ba04a9c980fedb91fd0901328.svg?invert_in_darkmode&sanitize=true" align=middle width=13.61499809999999pt height=26.085962100000025pt/>是one-hot向量，假设<img src="/tex/6a36c32ba04a9c980fedb91fd0901328.svg?invert_in_darkmode&sanitize=true" align=middle width=13.61499809999999pt height=26.085962100000025pt/>的真实标记为<img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>  
则：  
<p align="center"><img src="/tex/993ede9db595831bb98d78a5c5e10bcd.svg?invert_in_darkmode&sanitize=true" align=middle width=327.38211825pt height=42.03361965pt/></p>  
<p align="center"><img src="/tex/55ef77d802273c2f6f9ba07fb455ce0f.svg?invert_in_darkmode&sanitize=true" align=middle width=159.84953819999998pt height=42.03361965pt/></p>  
很容易得出：  
<p align="center"><img src="/tex/153f09c00ec2ddecc263fde77392c56e.svg?invert_in_darkmode&sanitize=true" align=middle width=255.79218554999997pt height=19.526994300000002pt/></p>  

很常见的结论，一定要知道的。  

当<img src="/tex/a86f6b2844f6175efb22e63a8f54b85a.svg?invert_in_darkmode&sanitize=true" align=middle width=85.3881567pt height=24.65753399999998pt/>时，随机选择能选对的概率为<img src="/tex/0b9171ea3b708b8f1dd476de06ac879e.svg?invert_in_darkmode&sanitize=true" align=middle width=77.02743344999999pt height=27.77565449999998pt/>，<img src="/tex/78e020638e18bf508b8f0ef1d43b3e7b.svg?invert_in_darkmode&sanitize=true" align=middle width=74.59424609999999pt height=22.831056599999986pt/>(困惑度)为<img src="/tex/dc7379d5fe8be9289709a7a9cf5176e1.svg?invert_in_darkmode&sanitize=true" align=middle width=83.58008339999999pt height=27.77565449999998pt/>。<img src="/tex/4f888b8a1c2bb650dff3ab83a22d690b.svg?invert_in_darkmode&sanitize=true" align=middle width=161.78872874999996pt height=22.831056599999986pt/>。  

![3b](Assignment2-img/3b.jpg)  
**解：**  
根据题目可知：<img src="/tex/a39b8d12f79be36b674b90d14d8c59a0.svg?invert_in_darkmode&sanitize=true" align=middle width=78.63405659999998pt height=29.190975000000005pt/>。  
现在设：  
<p align="center"><img src="/tex/0f8029ea7d94898c0e0a27d1f3190213.svg?invert_in_darkmode&sanitize=true" align=middle width=189.91805745pt height=17.883133949999998pt/></p>  
<p align="center"><img src="/tex/d5b2255520efa2e942eed42d02552dd2.svg?invert_in_darkmode&sanitize=true" align=middle width=118.40016705pt height=17.883133949999998pt/></p>  

则前向传播为：  
<p align="center"><img src="/tex/4d011d702643952080fb488942ce9b78.svg?invert_in_darkmode&sanitize=true" align=middle width=82.27741995pt height=15.4174053pt/></p>  
<p align="center"><img src="/tex/0f8029ea7d94898c0e0a27d1f3190213.svg?invert_in_darkmode&sanitize=true" align=middle width=189.91805745pt height=17.883133949999998pt/></p>  
<p align="center"><img src="/tex/f14813b88164bd219866e4724a9d0d7e.svg?invert_in_darkmode&sanitize=true" align=middle width=143.274846pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/d5b2255520efa2e942eed42d02552dd2.svg?invert_in_darkmode&sanitize=true" align=middle width=118.40016705pt height=17.883133949999998pt/></p>  
<p align="center"><img src="/tex/cf49ed11b501304a8d1b58f224d0b6bd.svg?invert_in_darkmode&sanitize=true" align=middle width=147.59349495pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/0e86a1cf8f9896ce13292b74be3f3933.svg?invert_in_darkmode&sanitize=true" align=middle width=144.19564995pt height=19.526994300000002pt/></p>  

反向传播：  
中间值：  
<p align="center"><img src="/tex/049b78327730219d6e8c54a279a96074.svg?invert_in_darkmode&sanitize=true" align=middle width=176.85982875pt height=37.2867957pt/></p>  
<p align="center"><img src="/tex/4b4918337ae0ebe129f23aa2990bcc09.svg?invert_in_darkmode&sanitize=true" align=middle width=513.16413225pt height=37.2867957pt/></p>  

则有：  
<p align="center"><img src="/tex/21d2b2a1ab92462684a1d0b54af9c3b0.svg?invert_in_darkmode&sanitize=true" align=middle width=195.41068305pt height=39.4577733pt/></p>  
<p align="center"><img src="/tex/ae6e27b46bfa7cfba0ffef9301f95359.svg?invert_in_darkmode&sanitize=true" align=middle width=282.0031929pt height=37.2867957pt/></p>  

<p align="center"><img src="/tex/f31387aa18a27206a96a17e11cda6e2d.svg?invert_in_darkmode&sanitize=true" align=middle width=263.35967625pt height=37.2867957pt/></p>  
<p align="center"><img src="/tex/1c273d41759fc74e52301c815d4aab69.svg?invert_in_darkmode&sanitize=true" align=middle width=295.86422415pt height=40.787721149999996pt/></p>  

<p align="center"><img src="/tex/b6241d35686252dfe3c6a61b3ee6ac0e.svg?invert_in_darkmode&sanitize=true" align=middle width=266.36363159999996pt height=37.2867957pt/></p>  

**如果你对上面的反向传播中的求导有疑惑，那么请看下面的简单讲解**  

考虑如下求导：  
<p align="center"><img src="/tex/8b251ef744562d971a7e16894b36b93e.svg?invert_in_darkmode&sanitize=true" align=middle width=194.334657pt height=36.2778141pt/></p>  

假设除了<img src="/tex/3df5ba8c81795df8b2cc1cd2347b50a2.svg?invert_in_darkmode&sanitize=true" align=middle width=15.182123549999996pt height=28.92634470000001pt/>，前面的已经求出了  
<p align="center"><img src="/tex/2acf32cbdda4360dbee34102794eea13.svg?invert_in_darkmode&sanitize=true" align=middle width=156.97279785pt height=36.2778141pt/></p>  

现在就差<img src="/tex/3df5ba8c81795df8b2cc1cd2347b50a2.svg?invert_in_darkmode&sanitize=true" align=middle width=15.182123549999996pt height=28.92634470000001pt/>了。需要讨论两种情况：  

1. 其中，<img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/>是一个行向量<img src="/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>乘上一个矩阵<img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/>，然后对矩阵<img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/>求导：  
<p align="center"><img src="/tex/c862bd9f6bb77a249d0e7c928899fc62.svg?invert_in_darkmode&sanitize=true" align=middle width=112.6490079pt height=33.81208709999999pt/></p>  

结果为<img src="/tex/1de6613435a641e2634d7340b551e189.svg?invert_in_darkmode&sanitize=true" align=middle width=17.406673349999988pt height=27.6567522pt/> **左乘** 前面一坨的求导结果<img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/>，即：  

<p align="center"><img src="/tex/aab3db138308a76bbd1370017eadd36b.svg?invert_in_darkmode&sanitize=true" align=middle width=82.2555789pt height=33.81208709999999pt/></p>  

而具体到题目中就是：  
<p align="center"><img src="/tex/6a80cae43c1476d8081280f4bb6eabdb.svg?invert_in_darkmode&sanitize=true" align=middle width=83.45658255pt height=37.2867957pt/></p>  
<p align="center"><img src="/tex/b4049fa20b98c4c89608f1f3544670b6.svg?invert_in_darkmode&sanitize=true" align=middle width=450.96802575000004pt height=36.99204465pt/></p>  
所以：  
<p align="center"><img src="/tex/5050d9b2fded8a901ea2cdb00d520f0a.svg?invert_in_darkmode&sanitize=true" align=middle width=282.0031929pt height=37.2867957pt/></p>  

2. 其中，<img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/>是一个行向量<img src="/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>乘上一个矩阵<img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/>，然后对行向量<img src="/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>求导：  

<p align="center"><img src="/tex/b6d0dcf502ea0ce7754147740d870c2e.svg?invert_in_darkmode&sanitize=true" align=middle width=102.78223229999999pt height=33.81208709999999pt/></p>  

结果为<img src="/tex/9f594b0805372992cdd22e5e7d40b67f.svg?invert_in_darkmode&sanitize=true" align=middle width=18.928705949999994pt height=27.6567522pt/> **右乘** 前面一坨的求导结果<img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/>，即：  

<p align="center"><img src="/tex/1b01e8348fbffb175c9ea76d02ad3d71.svg?invert_in_darkmode&sanitize=true" align=middle width=91.30046309999999pt height=33.81208709999999pt/></p>  

而具体到题目中就是：
<p align="center"><img src="/tex/b6241d35686252dfe3c6a61b3ee6ac0e.svg?invert_in_darkmode&sanitize=true" align=middle width=266.36363159999996pt height=37.2867957pt/></p>  


![3c](Assignment2-img/3c.jpg)
**解：**  
RNN的反向传播是按时间的反向传播，对于时间步<img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>的损失函数<img src="/tex/64124374b293d810758b19fec08e0c12.svg?invert_in_darkmode&sanitize=true" align=middle width=25.93617509999999pt height=29.190975000000005pt/>要沿时间向前传播，故现在为了方便，定义时间步<img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>的损失函数<img src="/tex/64124374b293d810758b19fec08e0c12.svg?invert_in_darkmode&sanitize=true" align=middle width=25.93617509999999pt height=29.190975000000005pt/>对每一时间步的误差项<img src="/tex/5e4ecfcce4eec96726a9e8067c469f48.svg?invert_in_darkmode&sanitize=true" align=middle width=23.167895849999987pt height=29.190975000000005pt/>:  
<p align="center"><img src="/tex/76a6a83baa21be33d38e09e81732cf27.svg?invert_in_darkmode&sanitize=true" align=middle width=84.278469pt height=37.2867957pt/></p>

现推导误差项的传播：  
<p align="center"><img src="/tex/cb7c09f11255c27df3fe4d3691ab5dad.svg?invert_in_darkmode&sanitize=true" align=middle width=192.76536675pt height=37.2867957pt/></p>  
<p align="center"><img src="/tex/a62b22e74d0bc096350ef14e2ceb08f2.svg?invert_in_darkmode&sanitize=true" align=middle width=262.8580812pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/8357ac5af1323eba8e6f7c09e7f10e2b.svg?invert_in_darkmode&sanitize=true" align=middle width=223.6350798pt height=37.2867957pt/></p>  
故可得递推式：  
<p align="center"><img src="/tex/0aea51d4180a4d566baf526e3c179d00.svg?invert_in_darkmode&sanitize=true" align=middle width=454.28160195pt height=37.2867957pt/></p>  
即可得：  
<p align="center"><img src="/tex/407e3ba0a2be653c2ee5429fbc15317c.svg?invert_in_darkmode&sanitize=true" align=middle width=461.91590114999997pt height=40.787721149999996pt/></p>  

<p align="center"><img src="/tex/83b60f5a4ada1f9fd234766a8cfcd8e5.svg?invert_in_darkmode&sanitize=true" align=middle width=533.5653014999999pt height=37.2867957pt/></p>  

<p align="center"><img src="/tex/017c5275dad2df9efa29dbf9d0e76b03.svg?invert_in_darkmode&sanitize=true" align=middle width=535.3822782pt height=37.2867957pt/></p>  


注意，上述过程用到了<img src="/tex/a1d10fd8a1c1198fcba88aa30b7107c3.svg?invert_in_darkmode&sanitize=true" align=middle width=58.41940499999999pt height=22.831056599999986pt/> 函数的导数:  
<p align="center"><img src="/tex/3242e0d108c2d780b55052e426ec78ac.svg?invert_in_darkmode&sanitize=true" align=middle width=179.64021735pt height=17.2895712pt/></p>  

![3d](Assignment2-img/3d.jpg)  
前向传播的复杂度分别为：  
<p align="center"><img src="/tex/ba58e475ecd51333d0c1c9d8bb4f7ca0.svg?invert_in_darkmode&sanitize=true" align=middle width=166.0491261pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/41a8ba930ba45c97ebeee90b7ece4eb6.svg?invert_in_darkmode&sanitize=true" align=middle width=350.82028245pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/605e1d4b185876d4a90c25b2a77887dd.svg?invert_in_darkmode&sanitize=true" align=middle width=226.79960984999997pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/a31b082c621984fc1e210a26ebee66ff.svg?invert_in_darkmode&sanitize=true" align=middle width=225.12132224999996pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/ebed795973774142432a2ba274076722.svg?invert_in_darkmode&sanitize=true" align=middle width=231.3652011pt height=19.526994300000002pt/></p>  
<p align="center"><img src="/tex/22084e90f705291737447a01219219ee.svg?invert_in_darkmode&sanitize=true" align=middle width=227.9673561pt height=19.526994300000002pt/></p>  
综上，在有两阶的时候则只保留两阶的情况下，前向传播的复杂度为：  

<p align="center"><img src="/tex/b72f3f1a5a8dc85bd18ba55131821223.svg?invert_in_darkmode&sanitize=true" align=middle width=163.2762945pt height=18.312383099999998pt/></p>  
同理，反向传播的复杂度为：  
<p align="center"><img src="/tex/b72f3f1a5a8dc85bd18ba55131821223.svg?invert_in_darkmode&sanitize=true" align=middle width=163.2762945pt height=18.312383099999998pt/></p>  

上述是第一个时间步长的复杂度，而<img src="/tex/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.046852649999991pt height=14.15524440000002pt/>个时间步的话就是：  

*  一次损失函数对<img src="/tex/7797f206bfea4f607bdbf5c74e364ba2.svg?invert_in_darkmode&sanitize=true" align=middle width=24.71093954999999pt height=29.190975000000005pt/>的求导，复杂度为<img src="/tex/3ecbce194b3027a2b45d8fc9059c459d.svg?invert_in_darkmode&sanitize=true" align=middle width=70.28287364999998pt height=24.65753399999998pt/>；
*  <img src="/tex/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.046852649999991pt height=14.15524440000002pt/>次反向传播，复杂度为<img src="/tex/2cb7e064c8a2915e24a73e61527c3204.svg?invert_in_darkmode&sanitize=true" align=middle width=120.51535485pt height=26.76175259999998pt/>;  
  
故，<img src="/tex/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.046852649999991pt height=14.15524440000002pt/>个时间步的反向传播复杂度为：  
<p align="center"><img src="/tex/2df80a695a8f87fd76a76416f22178b8.svg?invert_in_darkmode&sanitize=true" align=middle width=185.10856155pt height=18.312383099999998pt/></p>  

而如果是对前<img src="/tex/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.046852649999991pt height=14.15524440000002pt/>个词，每次都进行<img src="/tex/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode&sanitize=true" align=middle width=9.046852649999991pt height=14.15524440000002pt/>步的反向传播，那么复杂度大概为：  
<p align="center"><img src="/tex/2573bb7796992d098f99416eaf062fbe.svg?invert_in_darkmode&sanitize=true" align=middle width=201.5298549pt height=18.312383099999998pt/></p>  

