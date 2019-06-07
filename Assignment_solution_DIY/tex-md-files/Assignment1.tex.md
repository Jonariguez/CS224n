$$ Assignment\#1-solution $$  

![1a](Assignment1-img/1a.jpg)  
  
解答：
$$ softmax(\mathbf{x})_i=\frac{e^{x_i}}{\sum_{j}{e^{x_j}}}=\frac{e^ce^{x_i}}{e^c\sum_{j}{e^{x_j}}}=\frac{e^{x_i+c}}{\sum_{j}{e^{x_j+c}}}=softmax(\mathbf{x}+c)_i $$  
即  
$$ softmax(\mathbf{x})=softmax(\mathbf{x}+c) $$  
证毕  

![1b](Assignment1-img/1b.jpg)  