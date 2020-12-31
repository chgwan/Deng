# Introduction 

# Machine Learning Model

# Explanation of directory structure
1. getData 获取数据并且转换数据
2. MLModel 具体机器学习模型

# Some problems to solove.
1. 为什么 /scratch/chgwang/XI/data/1/1Normal/ia.mat 只有一条线 但是/scratch/chgwang/XI/data/1/2single/F1/IAF1.mat却有三条线，正常的均是三条线 所以一次数据是要储存三个信号一个时间轴。
2. 三条线是怎么画出来的？ **三相电所以是三根线**
3. 我们是不是要预测 每一个节点是否出现破坏**(我认为是这样)**, 可以类比成一个多分类问题.还是只预测fault是否出现?
4. 每个文件夹的名称的含义 **Get**
5. label distribution 是不是要均衡?
6. 什么样的采样率比较合适? **Get**
7. 周期是不是全都一致?如果一致的话周期是多少? 50Hz 0.02s 
8. 实际处理的时候是不是仅仅只需要稳定段,而不是所有的段都需要得到考虑? 一个周期?

https://github.com/chgwan/Deng

# Data analysis
1. 3道信号之间相互有影响，三项电流不能认为是完全独立的
2. 而且由于是周期性变化的，所以时间关系不是很重要
3. 那么根据CNN的用途，应该是可以采用CNN的，何不尝试一下GoogleNet
# 知识To Do.
1. LSTM RNN GRU (应该会选择LSTM)
2. Encoder-Decoder model
3. Encoder-Decoder model + Attention 
4. [知乎介绍](https://zhuanlan.zhihu.com/p/91839581)
5. [文章参考](https://arxiv.org/abs/2007.10552)