# Lyrics-generation

歌词生成，包括数据爬取、数据处理、模型训练和歌词生成，主要使用了基于lstm的语言模型和基于seq2seq的序列模型进行训练。

## 一、文件说明

- data_crawl: 数据爬取文件

- data_processing：输出处理文件

- lstm_model：基于lstm的概率语言模型进行歌词生成，train_lstm_word_based.py是训练代码，generateLyrics_word_based.py是生成代码

- seq2seq_model：基于seq2seq序列模型进行歌词生成，train_seq2seq_word_based.py是训练代码，generation_word_based.py是生成代码

- train_data：训练数据

- 训练好的参数和word2index以及index2word文件：https://pan.baidu.com/s/1Nx7bEIja-VqJv83kT6tpAg

## 二、使用说明

### 1、训练数据

```
# lstm模型训练
> cd lstm_model
> python train_lstm_word_based.py

# seq2seq模型训练
> cd seq2seq_model
> python train_seq2seq_word_based.py
```

### 2、生成歌词
```

# 将训练好的参数或者百度云盘上下载的文件放入相应文件夹下

# lstm模型生成歌词
> cd lstm_model
> python generateLyrics_word_based.py

# seq2seq模型训练
> cd seq2seq_model
> python generation_word_based.py
```

## 三、详细说明

### 1、数据爬取
主要爬取网易云音乐网页（https://music.163.com） 的歌词数据。

分为以下三步骤：

**Step1：** 获取歌手专辑id信息

**Step2：** 根据专辑id获取这张专辑中包含的歌曲id

**Step3：** 根据歌曲id爬取歌词

具体的代码在data_crawl文件夹中。

### 2、数据处理

数据处理主要将数据爬取中的结果（多个歌曲文件）合并成一个文件，并去掉一些歌词生成的干扰信息，如歌手：李荣浩，作词：李荣浩等信息，另外，本任务是生成中文歌词，可以将一些英文歌词去掉。

### 3、歌词生成

歌词生成主要使用了2种模型：

- 基于lstm的概率语言模型
- 基于seq2seq的序列模型

下面大体介绍下两种模型的情况（具体的内容请移步本人博客：https://blog.csdn.net/quiet_girl/article/details/84768821）：

#### 3.1 基于lstm的概率语言模型

（1）模型结构如下：

<div align=center><img src="https://github.com/Nana0606/Lyrics-generation/blob/master/imgs/lstm_structure.png" width="50%" alt="基于lstm的概率语言模型结构"/></div>

本模型使用了2层LSTM，一层全连接层，全连接层后面接了一个softmax，也就是分类模型。

（2）模型训练结果：

<div align=center><img src="https://github.com/Nana0606/Lyrics-generation/blob/master/imgs/lstm_train_analysis.png" width="60%" alt="基于lstm的概率语言模型训练结果"/></div>

从上图可以看出，当loss增加的时候，accuracy降低，在训练的时候，大概10个epoch左右就已经达到最好的状态。

（3）结果展示：

A、以“喧嚣的人群”开头

```
喧嚣的人群恩法
到底我多事全仅诉
我真的不想吗
我不想就想你走不放
孤独觉得心乱
空虚暖通
不要却把回忆抹开
其实还有什么是真理
茫风叫相恭潮自我
如果说分手
等和爱和谁幸福的脸
寒流来深月
开对在多年轻欢
我用还在一起
不分的问题
没有看起来
可惜总是很难受知道
我在你的时候
你要离开这醉不铃
这是我写的歌
曾经如何真
如同的波延
念了大师们在失着梦里里
让每个爱你
难以用爱
有分开是年轻
将你
```

B、以“痴情”开头

```
痴情
天台月光希
他们它真的一点人
人世漂代傻
他背纪那一刀的晚表
你留着本绪路的车和能发现
有话就该更习惯
真的够小关还
不代一样的激悟
你家起温暖的大岛
每一众想要在一颗
想点一非往前走的都走
月下窗祖滴咸嘣该跟你无老
哦
他还是注想
我会拉失下白后
想总是为了你
要不是忍到何不民
你明了的温摸
炫散
有就进
都爱了
不代表骤什么
生命
名瑟的是谁
美女诉那对情都好
英雄流浪的小啦
隐约远的孤
```

C、以“自由”开头

```
自由走
我为你方开
做了几个甜微的火尘世那几年
苦笑都不自客
人生已远离这
我看不到
听说在界缘里捉反坠
西弱过的白坤
我们肢阔都
坚持多情之前
一张居怕非醉
恋情
我牵着你的手
我像狗草月下
世幕轻嚓现路唱了被左都市
拉影后魂要用意义塌伪
靠告诉我们睡过的也没有发色这
有的航点
不对这失情的雨气
坚强自在也事我
被放手的公园而片
落水跟肉家
烟雾里往模知
莹天上电影
海漠就该灯地
淡淡事起民视残
```

#### 3.2 基于seq2seq的序列模型

（1）模型结构如下：

<div align=center><img src="https://github.com/Nana0606/Lyrics-generation/blob/master/imgs/seq2seq_structure.png" width="60%" alt="基于seq2seq的序列模型结构"/></div>

decoder和encoder部分都是基于lstm，在decoder后面有一层dense层，之后又接了一个softmax层，用于预测。

（2）模型训练结果：

<div align=center><img src="https://github.com/Nana0606/Lyrics-generation/blob/master/imgs/seq2seq_train_analysis.png" width="60%" alt="基于lstm的概率语言模型训练结果"/></div>

从上图可以看出，训练集的acc还在持续增加，loss持续降低；但是验证集上val-loss在先减小再减小（尽管val-acc一直在增加），这是典型的过拟合（下篇博文会整理出现这个问题的原因），但是我们存储的val-loss最小时的模型，在epoch=15，val-loss≈3.1，val-acc≈0.45。（具体详见原因分析详见博客）

（3）结果展示：

A、以“喧嚣的人群”开头

```
喧嚣的人群
别让我受伤悲哀
可以
折拿
效
发谈
根皮
满
下根
道
根
```

B、以“痴情”开头

```
痴情
晚到
更的与片
千彼生活成
段
子有大海
泥遍
花
甘台仙仙
度
拼下泥
```

C、以“自由的人儿”开头

```
自由
我决定
生鱼的浪旁
烂
泥术羿回光
射下讨
光亮亮
角角有角空
予回起角
度
光角度限
```

## 4、总结

从上面可以看出，基于概率语言模型的生成效果好于基于序列模型，可能有以下原因：

A、基于序列的模型主要对有前后关联信息的上下文有利，而歌词和诗歌还有些不同，诗歌的前后关联更强一些，而歌词很多上下句是关联不大的，如下面的歌词（罗大佑《倒影》中的歌词）：

```
会吗会吗
只因为软弱的自己
总不能说我不理
要嘛要嘛
```

B、对于序列模型这里因为资源问题没有进行调参，在模型不断训练的过程中，很快出现了过拟合，如果采取一些方式，可能效果更好。

C、在seq2seq模型中，最开始使用一个歌手的数据，结果非常差，后来扩大了数据集，效果好很多，想让结果更好，可以考虑继续增大训练集。

## 四、参考文章

主要参考Keras官方examples。
