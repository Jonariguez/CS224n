<p align="center"><img src="/tex/981e1a911d5d015cd4e9ca83bdef1a5e.svg?invert_in_darkmode&sanitize=true" align=middle width=191.34590144999999pt height=14.611878599999999pt/></p>  

![1a](Assignment1-img/1a.jpg)  
  
解答：
<p align="center"><img src="/tex/35995193a6c6128f08c9bd37b6e628b8.svg?invert_in_darkmode&sanitize=true" align=middle width=490.57258634999994pt height=42.65424074999999pt/></p>  
即  
<p align="center"><img src="/tex/62b524786a05e90fb15e7ea2c7959f36.svg?invert_in_darkmode&sanitize=true" align=middle width=222.53629034999997pt height=16.438356pt/></p>  
证毕  

![1b](Assignment1-img/1b.jpg)  

解答：  
直接在代码中利用numpy实现即可。注意要先从<img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>中减去每一行的最大值，这样在保证结果不变的情况下，所有的元素不大于0，不会出现`上溢出`，从而保证结果的正确性。具体可参考 http://www.hankcs.com/ml/computing-log-sum-exp.html  
  
![2a](Assignment1-img/2a.jpg)   

解答：
<p align="center"><img src="/tex/db9928d4fd9b64284c4f1a6d42ba3cc0.svg?invert_in_darkmode&sanitize=true" align=middle width=427.2343086pt height=39.59480249999999pt/></p>  

即<img src="/tex/a1d10fd8a1c1198fcba88aa30b7107c3.svg?invert_in_darkmode&sanitize=true" align=middle width=58.41940499999999pt height=22.831056599999986pt/>函数的求导可以由其本身来表示。

![2b](Assignment1-img/2b.jpg)  

解答：  
我们知道真实标记<img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>是one-hot向量，因此我们下面的推导都基于 <img src="/tex/d188eb9ee9e3183afbdec8b468d271bf.svg?invert_in_darkmode&sanitize=true" align=middle width=46.28421599999999pt height=21.18721440000001pt/> ,且 <img src="/tex/92fc33d17362605fd0aaa5e37d91e79f.svg?invert_in_darkmode&sanitize=true" align=middle width=87.63117659999999pt height=22.831056599999986pt/> ，即真实标记是 <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> .  

<p align="center"><img src="/tex/88ef3df5488b0c66f5cf98311d9cf84d.svg?invert_in_darkmode&sanitize=true" align=middle width=206.7417165pt height=37.9216761pt/></p>  

其中：  
<p align="center"><img src="/tex/cae0e852e541c146d64bfa168bd0ade4.svg?invert_in_darkmode&sanitize=true" align=middle width=209.23667655pt height=42.84447255pt/></p>  

接下来讨论 <img src="/tex/d29eac8ed44d78c9fe9b67a26afabebc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.251585249999996pt height=30.648287999999997pt/> :  
1) <img src="/tex/adb954c91e658590185031beeb908ca2.svg?invert_in_darkmode&sanitize=true" align=middle width=36.65622509999999pt height=22.831056599999986pt/>:  
<p align="center"><img src="/tex/9b42f0140270a2e17adc2c751b84ff0f.svg?invert_in_darkmode&sanitize=true" align=middle width=250.53299205pt height=43.521805799999996pt/></p>  

则：  
<p align="center"><img src="/tex/069026df3d257f09b250f873c8e2ecd8.svg?invert_in_darkmode&sanitize=true" align=middle width=337.05503534999997pt height=37.0084374pt/></p>

2) <img src="/tex/4825b4d7040a1b0ab62a9ee3331b7287.svg?invert_in_darkmode&sanitize=true" align=middle width=36.65622509999999pt height=22.831056599999986pt/>:  
<p align="center"><img src="/tex/48d03e3246e339d8124318b30d450715.svg?invert_in_darkmode&sanitize=true" align=middle width=213.55523969999996pt height=43.521805799999996pt/></p>  

则：  
<p align="center"><img src="/tex/c011b1357fbd33e66ed3842137d01431.svg?invert_in_darkmode&sanitize=true" align=middle width=289.78262939999996pt height=37.0084374pt/></p>  

综上：  
<p align="center"><img src="/tex/654ac2729bd140acdb9b0426bf1a0691.svg?invert_in_darkmode&sanitize=true" align=middle width=205.1064147pt height=49.315569599999996pt/></p>  

或者：

<p align="center"><img src="/tex/dcc4e9cbbbb091e8594bdd266757d597.svg?invert_in_darkmode&sanitize=true" align=middle width=134.31675345pt height=37.1910528pt/></p>  

![2c](Assignment1-img/2c.jpg)  

解答：  
首先设：<img src="/tex/7ae7b79130069ec6f6ab73744840a29a.svg?invert_in_darkmode&sanitize=true" align=middle width=106.50668654999998pt height=22.831056599999986pt/> 和 <img src="/tex/cc62bf57a815668bdb239e4a0ed71b92.svg?invert_in_darkmode&sanitize=true" align=middle width=106.58281424999998pt height=22.831056599999986pt/>，那么前向传播的顺序依次为：  

<p align="center"><img src="/tex/b4ed13c4a160555c336141297985afbe.svg?invert_in_darkmode&sanitize=true" align=middle width=106.50668655pt height=13.881256950000001pt/></p>
<p align="center"><img src="/tex/287023cd062b28101901deca995ec2e0.svg?invert_in_darkmode&sanitize=true" align=middle width=121.18952339999998pt height=16.438356pt/></p>
<p align="center"><img src="/tex/30fafd31a89230080c2fbe61fb7606fc.svg?invert_in_darkmode&sanitize=true" align=middle width=106.58281425pt height=13.881256950000001pt/></p>
<p align="center"><img src="/tex/1121619c14f5e30af0b96bf02f2580bf.svg?invert_in_darkmode&sanitize=true" align=middle width=125.89248705pt height=16.438356pt/></p>
<p align="center"><img src="/tex/02fe4132c041c5015951a02fdacd6a83.svg?invert_in_darkmode&sanitize=true" align=middle width=221.41387619999998pt height=36.6554298pt/></p>

现在求<img src="/tex/3907d1fcf323c95464418c95ef0d1437.svg?invert_in_darkmode&sanitize=true" align=middle width=16.086209699999998pt height=28.92634470000001pt/>,其实就是进行一次反向传播：  
<p align="center"><img src="/tex/393336cd04583e9fb7f45894278fd735.svg?invert_in_darkmode&sanitize=true" align=middle width=128.08681545pt height=36.2778141pt/></p>
<p align="center"><img src="/tex/c5bfaf6dbc0f5f48a25e535e752e154d.svg?invert_in_darkmode&sanitize=true" align=middle width=375.26687384999997pt height=36.2778141pt/></p>
<p align="center"><img src="/tex/f6bad702a8f943db838dc1265181956f.svg?invert_in_darkmode&sanitize=true" align=middle width=375.75722579999996pt height=37.1910528pt/></p>
<p align="center"><img src="/tex/91de91ed4fdc3d8790d5befe97938e51.svg?invert_in_darkmode&sanitize=true" align=middle width=435.43210094999995pt height=36.2778141pt/></p>

![2d](Assignment1-img/2d.jpg)  

解答：  
(1) 从输入层到隐藏层，全连接共<img src="/tex/0373fd6e72358d8b50f11e8b30d66f22.svg?invert_in_darkmode&sanitize=true" align=middle width=56.97704429999999pt height=22.465723500000017pt/>个，即<img src="/tex/4c0c82cdc5d7bf2312fe6669d3f632f3.svg?invert_in_darkmode&sanitize=true" align=middle width=22.07767979999999pt height=22.465723500000017pt/>，加上<img src="/tex/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>个偏置项，共<img src="/tex/47df53dfe90e0bbe08fda80bcf0aa1f8.svg?invert_in_darkmode&sanitize=true" align=middle width=92.06820479999999pt height=22.465723500000017pt/>个。  
(2) 从隐藏层到输出层，共<img src="/tex/b4f3dae50d17523f3305e04419370242.svg?invert_in_darkmode&sanitize=true" align=middle width=97.38265349999999pt height=22.465723500000017pt/>个。  
参数个数共：
<p align="center"><img src="/tex/c3a6ccee4af0503fad9a9f335684b50b.svg?invert_in_darkmode&sanitize=true" align=middle width=235.93478864999997pt height=17.031940199999998pt/></p>

![2e](Assignment1-img/2e.jpg)  

![2f](Assignment1-img/2f.jpg)  

![2g](Assignment1-img/2g.jpg)  