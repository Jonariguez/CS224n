# CS224n
Stanford CS224n Natural Language Processing with Deep Learning课程资料(**2017**)  
课程官网 http://web.stanford.edu/class/cs224n/  
包括课件([lecture_slides](lecture_slides/))、讲义([lecture_notes](lecture_notes/))、作业(assigment)及作业解答。

## 依赖
* python 2.7
* tensorflow 0.12.1
  
## 指南
* `Assignment`等文件夹中是对应的作业，包括作业的`pdf`，`代码`，`.sh`文件等等，**它们都是原始的，即没有任何标记，代码里没有任何已实现的代码，这样可以方便大家有需要的可以下载下来自己去研究、去独立完成**。  
* 我自己的完成的作业和实现的代码，都在`Assignment_solution_DIY`文件夹下，只是作为备忘或者笔记保存，大家也可以参考或者指正错误，多谢指教。  
* 我自己完成的代码中，`Assignment1`的运行环境是python2.7，`Assignment2`和`Assignment3`是python3.6  

## 问题
1. Assignment1中的通过运行`get_datasets.sh`脚本不能成功获得数据  
首先在`Assignment1\utils`文件夹下创建`datasets`文件夹，该文件夹下应有两份数据：  
   * `stanfordSentimentTreebank`文件夹  
    获取方式为 http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip ，将下载的压缩包解压之后的文件夹放在`Assignment1\utils\datasets`文件夹下即可。
    * `glove.6B.50d.txt`  
    获取方式为 http://nlp.stanford.edu/data/glove.6B.zip ，将下载的压缩包解压之后,将其中的`glove.6B.50d.txt`放在`Assignment1\utils\datasets`文件夹下即可。    

2. 在`Assignment3`的`util.py`文件`import StringIO`报错  
   因为后面两次作业的代码我是用的**python3.6**，所以会遇到这样的错误。在**python3**中已经把`StringIO`放到了`io`模块中，改成`from io import StringIO`即可。

## 学习资料推荐
* 码农场CS224n课程学习笔记  
http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html  
* Github中.md文件的LaTex渲染  
https://stackoverflow.com/questions/35498525/latex-rendering-in-readme-md-on-github  
* 在线LaTex公式编辑器  
http://latex.91maths.com/  
https://www.codecogs.com/latex/eqneditor.php  
* 利用.gitignore忽略本地和已提交的文件  
https://www.cnblogs.com/YangJieCheng/p/8175761.html
* python 3.6 TypeError: write() argument must be str, not bytes解决办法  
  https://www.jianshu.com/p/b53ac2effb85