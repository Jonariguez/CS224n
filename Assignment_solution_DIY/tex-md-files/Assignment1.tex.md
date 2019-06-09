$$ Assignment\#1-solution\quad By\; Jonariguez $$  
  
**所有的代码题目对应的代码可查看对应文件夹AssignmentX_Code下的.py文件**
  
![1a](Assignment1-img/1a.jpg)  

**解：**
$$ softmax(\mathbf{x})_i=\frac{e^{x_i}}{\sum_{j}{e^{x_j}}}=\frac{e^ce^{x_i}}{e^c\sum_{j}{e^{x_j}}}=\frac{e^{x_i+c}}{\sum_{j}{e^{x_j+c}}}=softmax(\mathbf{x}+c)_i $$  
即  
$$ softmax(\mathbf{x})=softmax(\mathbf{x}+c) $$  
证毕  

![1b](Assignment1-img/1b.jpg)  

**解：**  
直接在代码中利用numpy实现即可。注意要先从$x$中减去每一行的最大值，这样在保证结果不变的情况下，所有的元素不大于0，不会出现`上溢出`，从而保证结果的正确性。具体可参考 http://www.hankcs.com/ml/computing-log-sum-exp.html  
  
![2a](Assignment1-img/2a.jpg)   

**解：**
$$ \sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\sigma(x)\cdot(1-\sigma(x)) $$  

即$sigmoid$函数的求导可以由其本身来表示。

![2b](Assignment1-img/2b.jpg)  

**解：**  
我们知道真实标记$y$是one-hot向量，因此我们下面的推导都基于 $y_k=1$ ,且 $y_i=0,i\neq k$ ，即真实标记是 $k$ .  

$$ \frac{\partial CE(y,\hat{y})}{\partial\theta}=\frac{\partial CE(y,\hat{y})}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial\theta} $$  

其中：  
$$ \frac{\partial CE(y,\hat{y})}{\partial\hat{y}}=-\sum_{i}{\frac{y_i}{\hat{y}_i}}=-\frac{1}{\hat{y}_k} $$  

接下来讨论 $\frac{\partial\hat{y}}{\partial\theta}$ :  
1) $i=k$:  
$$ \frac{\partial\hat{y}}{\partial\theta_k}=\frac{\partial}{\partial\theta_k}(\frac{e^{\theta_k}}{\sum_{j}{e^{\theta_j}}})=\hat{y}_k\cdot(1-\hat{y}_k) $$  

则：  
$$ \frac{\partial CE}{\theta_i}=\frac{\partial CE}{\partial\hat{y}}\frac{\partial\hat{y}}{\theta_i}=-\frac{1}{\hat{y}_k}\cdot\hat{y}_k\cdot(1-\hat{y}_k)=\hat{y}_i-1 $$

2) $i \neq k$:  
$$ \frac{\partial\hat{y}}{\partial\theta_i}=\frac{\partial}{\partial\theta_i}(\frac{e^{\theta_k}}{\sum_{j}{e^{\theta_j}}})=-\hat{y}_i\cdot\hat{y}_k $$  

则：  
$$ \frac{\partial CE}{\theta_i}=\frac{\partial CE}{\partial\hat{y}}\frac{\partial\hat{y}}{\theta_i}=-\frac{1}{\hat{y}_k}\cdot(-\hat{y}_i\cdot\hat{y}_k)=\hat{y}_i $$  

综上：  
$$ \frac{\partial CE(y,\hat{y})}{\partial\theta_i}=\begin{cases} \hat{y}_i-1 & i=k \\ \hat{y}_i & i\neq k \end{cases} $$  

或者：

$$ \frac{\partial CE(y,\hat{y})}{\partial\theta_i}=\hat{y}-y $$  

![2c](Assignment1-img/2c.jpg)  

**解：**  
首先设：$Z_1=xW_1+b_1$ 和 $Z_2=hW_2+b_2$，那么前向传播的顺序依次为：  

$$ Z_1=xW_1+b_1 $$
$$ h=sigmoid(Z_1) $$
$$ Z_2=hW_2+b_2 $$
$$ \hat{y}=softmax(Z_2) $$
$$ J=CE(y,\hat{y})=-\sum_{i}{y_ilog(\hat{y}_i)} $$

现在求$\frac{\partial J}{\partial x}$其实就是进行一次反向传播：  
$$ \delta_1 =\frac{\partial J}{\partial Z_2}=\hat{y}-y $$
$$ \delta_2 =\frac{\partial J}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial h}=(\hat{y}-y)\cdot\frac{\partial}{\partial x}(hW_2+b_2)=\delta_1\cdot W_2^T $$
$$ \delta_3 =\frac{\partial J}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial h}\cdot\frac{\partial h}{\partial Z_1}=\delta_2\cdot\frac{\partial (\sigma(Z_1))}{\partial Z_1}=\delta_2\odot\sigma'(Z_1) $$
$$ \frac{\partial J}{\partial x} =\frac{\partial J}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial h}\cdot\frac{\partial h}{\partial Z_1}\cdot\frac{\partial Z_1}{\partial x}=\delta_3\cdot\frac{\partial }{\partial x}(xW_1+b_1)=\delta_3\cdot W_1^T $$

![2d](Assignment1-img/2d.jpg)  

**解：**  
(1) 从输入层到隐藏层，全连接共$D_x\times H$个，即$W_1$，加上$H$个偏置项，共$D_x\times H+H$个。  
(2) 从隐藏层到输出层，共$H\times D_y+D_y$个。  
参数个数共：
$$ (D_x\times H+H)+(H\times D_y+D_y) $$

![2e](Assignment1-img/2e.jpg)  

![2f](Assignment1-img/2f.jpg)  

![2g](Assignment1-img/2g.jpg)  

![3a](Assignment1-img/3a.jpg)  

**解：**  
首先分析各个量的形状：$U=[u_1,u_2,...,u_W]\in d\times W$，$y,\hat{y}\in W\times 1$，其中$W$为词典大小，$d$为词向量的维度。   
我们设：  
$$ \theta=\begin{bmatrix}
u_1^Tv_c\\ 
u_2^Tv_c\\ 
...\\ 
u_W^Tv_c
\end{bmatrix}=U^Tv_c \in W\times 1$$  

则：  
$$ \hat{y}_o = p(o|c)=\frac{exp(u_o^Tv_c)}{\sum_{w=1}^{W}{exp(u_w^Tv_c)}}=softmax(\theta)_o $$  
$$ \hat{y} =softmax(\theta) $$  
那么：  
$$ \frac{\partial J}{\partial v_c}=\frac{\partial J}{\partial \theta}\cdot\frac{\partial \theta}{\partial v_c}=(\hat{y}-y)\cdot\frac{\partial }{\partial v_c}(U^Tv_c)=U(\hat{y}-y) $$  

![3b](Assignment1-img/3b.jpg)  

**解：**  
可以先对$U^T$求导：  
$$ \frac{\partial J}{\partial U^T}=\frac{\partial J}{\partial \theta}\cdot\frac{\partial \theta}{\partial U^T}=\frac{\partial J}{\partial \theta}\cdot\frac{\partial }{\partial U^T}(U^Tv_c)=(\hat{y}-y)\cdot v_c^T $$  
  
那么对 $U$ 求导的结果对上式转置即可：  

$$ \frac{\partial J}{\partial U}= ((\hat{y}-y)\cdot v_c^T)^T=v_c\cdot(\hat{y}-y)^T$$  
也可以表示为：  
$$ \frac{\partial J}{\partial U}=\begin{cases} (\hat{y}_w-1)\cdot v_c & w=o \\ \hat{y}_w\cdot v_c & w\neq o \end{cases} $$  

![3c](Assignment1-img/3c.jpg)  

**解：**  
首先应该知道：  
$$ \sigma'(x)=\sigma(x)\cdot(1-\sigma(x)) $$  
$$ 1-\sigma(x)=\sigma(-x) $$
已知：  
$$ J(o,v_c,U)=-log(\sigma(u_o^Tv_c))-\sum_{k=1}^{K}{log(\sigma(-u_k^Tv_c))} $$  

直接求导即可：  
$$ \frac{\partial J}{\partial v_c}=-\frac{\sigma'(u_o^Tv_c)\cdot u_o}{\sigma(u_o^Tv_c)}+\sum_{k=1}^{K}{\frac{\sigma'(-u_k^Tv_c)\cdot u_k}{\sigma(-u_k^Tv_c)}}= (\sigma(u_o^Tv_c)-1)u_o+\sum_{k=1}^{K}{\sigma(u_k^Tv_c)\cdot u_k}$$  
$$ \frac{\partial J}{\partial u_k}=\begin{cases}(\sigma(u_o^Tv_c)-1)v_c & k=o \\ \sigma(u_k^Tv_c)v_c & k\neq o \end{cases}  $$  

![3d](Assignment1-img/3d.jpg)

**解：**  
根据题目的提示可知，我们可以设$F(o,v_c)$为损失函数，等价于前面的$J_{softmax-CE}$或者$J_{neg-sample}$，而$J$对变量的求导我们前面已经做过，所以这里直接使用$\frac{\partial F(o,v_c)}{\partial ..}$代替即可，不用再进一步求导展开。  
(1) **skip-gram模型**  
$$ J_{skip-gram}(word_{c-m..c+m})=\sum_{-m\leq j\leq m,j\neq 0}{F(w_{c+j},v_c)} $$
$$ \frac{\partial J}{\partial U}=\sum_{-m\leq j\leq m,j\neq 0}{\frac{\partial F(w_{c+j},v_c)}{\partial U}} $$  
$$ \frac{\partial J}{\partial v_c}=\sum_{-m\leq j\leq m,j\neq 0}{\frac{\partial F(w_{c+j},v_c)}{\partial v_c}} $$   
$$ \frac{\partial J}{\partial v_j}=\vec{0}, j\neq c $$   
   
\(2) **CBOW模型**  
因为CBOW模型是根据多个背景词预测一个中心词，又因为$F()$惩罚函数是形如`(一个词，一个词)`的形式，所以要把多个背景词变成一个词，那么一种有效的方式就是把这些背景词的词向量求平均便得到了一个词向量。  
$$ \hat{v}=\sum_{-m\leq j\leq m,j\neq 0}{v_{c+j}} $$
$$ J_{CBOW}(word_{c-m..c+m})=F(w_c,\hat{v}) $$  
那么：  
$$ \frac{\partial J}{\partial U}=\frac{\partial F(w_c,\hat{v})}{\partial U} $$  
$$ \frac{\partial J}{\partial v_c}=\vec{0}, c\notin \{c-m,..,c-1,c+1,..c+m\} $$  
$$ \frac{\partial J}{\partial v_j}=\frac{\partial F(w_c,\hat{v})}{\partial \hat{v}}\cdot\frac{\partial \hat{v}}{\partial v_j}=\frac{\partial F(w_c,\hat{v})}{\partial v_j}, c\in \{c-m,..,c-1,c+1,..c+m\} $$  

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