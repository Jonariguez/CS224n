$$ Assignment\#1 -solution\quad By\ Jonariguez$$  

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
TensorFlow的自动梯度，是指我们使用时只需要定义图的就好了，不用自己实现自动梯度，反向传播和求导由TensorFlow自动完成。  
  

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
* 每个次都要进入stack中，故要有n步SHIFT操作。  
* 最终stack中只剩ROOT，即每一次ARC会从stack中删掉一个词，故共有n步LEFT-ARC和RIGHT-ARC操作。  

   
![2c](Assignment2-img/2c.jpg)  

![2d](Assignment2-img/2d.jpg)  

![21](Assignment2-img/21.jpg)  

![2e](Assignment2-img/2e.jpg)

![2f](Assignment2-img/2f.jpg)
**解：**
$$ \mathbb{E}_{p_{drop}}[\mathbf{h}_{drop}]=\mathbb{E}_{p_{drop}}[\gamma \mathbf{d}\circ \mathbf{h}]=p_{drop}\cdot \vec{0}+(1-p_{drop})\cdot\gamma\cdot\mathbf{h}=\mathbf{h} $$  
即推导出：  
$$ \gamma=\frac{1}{1-p_{drop}} $$  

![2f](Assignment2-img/2f.jpg)  

![2g1](Assignment2-img/2g1.jpg)  

**解:**  
因为其实$\mathbf{m}$是之前全部梯度(更新量)的加权平均，更能体现梯度的整体变化。因为这样减小了更新量的方差，避免了梯度振荡。  
$\mathbf{\beta_1}$一般要接近1。  

![2g2](Assignment2-img/2g2.jpg)  
**解：**  
* 更新量$\mathbf{m}$: 对梯度(更新量)进行滑动平均  
* 学习率$\mathbf{v}$: 对梯度的平方进行滑动平均  

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
由题目可知，$x^{(t)}$是one-hot**行向量**，且隐藏层也是行向量的形式。