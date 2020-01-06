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

下面大体介绍下两种模型的情况（具体的内容请移步本人博客：https://blog.csdn.net/quiet_girl/article/details/84768821 ）：

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
喧嚣的人群
单内时化的约影
隐血在约学
城朝笔连别伊气错
狗吃阿辉仔西饲
一个阿仔仔嘟色
来喝老伤情不管眼睛
飞得阳色的一通深得到
我的书像南才是一样
有一个梦的理车
在家停倾萄
悔给你的爱人
请开气吧
我靠不要岁下比防民
你的美豪必处发
每天不知从意紧每个人都好
好笑心心静却怎么偷也不存变
照了土玫笼的麦道
只来蚂印
微少那万秋月的门上
江倒院角飞
这目光
辉光期处拳在破上
陪你终于失去
雨成秋
```

B、以“痴情”开头

```
痴情的冲角
却走出时间很嚣
有此有想像一年
铁科铁岸的笔河节某坡入灯
缓睡如灯分如果
生著弟
大古悲是那此的黄空
护着后还已不机
等到心世将你
眼泪运内一个
你说你已不该归为爱在我写的歌
只像我早跳了岁三山字等
粉凡的地方
温作的时间
今眼已那一纪到
往下血为
你成了却幸福依险
写情做难来
同难需要做话
去走就这样乡呢没关来
为现来总有叶句正临到上要证望的唱时纷去
故样照去
几夜落清
忘了有些什么
```

C、以“自由”开头

```
自由
过去有几靠
接了时泪
却想收声
想起我却不想
不曾一样的俗物
爱过你
不是在这一个一个梦里
热处海望海间好头
越不可限当月
让平持潮灭
左心无影的结局
随你本心
一直到于你的情流我开始
还是那么由过了
原来我回忆满去去来
再谢你已买化里的意录
洋颗变园的音声境凉
每天却将都一起
你怀忘
这旧古鲜的属于大街
椅作页咧念出熟
死了青角
春起成开台折哭红的末脸
为些母冰光的围契平个很察以答展
我们
```

#### 3.2 基于seq2seq的序列模型

（1）模型结构如下：

<div align=center><img src="https://github.com/Nana0606/Lyrics-generation/blob/master/imgs/seq2seq_structure.png" width="60%" alt="基于seq2seq的序列模型结构"/></div>

decoder和encoder部分都是基于lstm，在decoder后面有一层dense层，之后又接了一个softmax层，用于预测。

（2）模型训练结果：

<div align=center><img src="https://github.com/Nana0606/Lyrics-generation/blob/master/imgs/seq2seq_train_analysis.png" width="60%" alt="基于lstm的概率语言模型训练结果"/></div>

从上图可以看出，训练集的acc还在持续增加，loss持续降低；但是验证集上val-loss在先减小再减小（尽管val-acc一直在增加），这是典型的过拟合（分析详见博客：https://blog.csdn.net/quiet_girl/article/details/86138489 ），但是我们存储的val-loss最小时的模型，在epoch=15，val-loss≈3.1，val-acc≈0.45。

（3）结果展示：

A、以“喧嚣的人群”开头

```
喧嚣的人群
别让我受伤悲哀
可以让我们的你
折拿的受伤了
效群的人们的受伤
发谈话喧哗嘈杂声
根皮树林春天来了
满满人群人们
下根本缺成为了
道路失去了终于
根本找到了
```

B、以“痴情”开头

```
痴情
晚到秋天
更的与片谈话
千彼生活成
段浪漫你我我们
子有大海浪漫
泥遍我爱你爱
花香啦啦啦
甘台仙仙
爱情你说呢
拼下泥的我们
```

C、以“自由的人儿”开头

```
自由
我决定成了
生鱼的浪旁
烂漫天堂
泥术羿回光
射下讨春光
光亮亮
角角有角空
予回起角
度情心情
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

B、对于序列模型这里因为没有采取过多的方式进行控制，在模型不断训练的过程中，很快出现了过拟合，如果采取一些方式，可能效果更好。

C、在seq2seq模型中，最开始使用一个歌手的数据，结果非常差，后来扩大了数据集，效果好很多，想让结果更好，可以考虑继续增大训练集。

## 四、参考文章

主要参考Keras官方examples。
