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
