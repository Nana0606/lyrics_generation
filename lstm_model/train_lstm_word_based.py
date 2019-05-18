# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/12/2 13:37

"""
本文主要使用把歌词生成当做分类任务来做，使用2层lstm+1层dense，最后使用softmax。
代码包括模型及参数的保存以及tensorboard-log的保存。
参考：https://github.com/shiwusong/keras_lstm_generation/blob/master/main.py#L24
"""

import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import json
import os

# 指定GPU和最大占用的显存比例
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

SEQ_LENGTH = 10     # 通过前n个词，生成后一个词，SEQ_LENGTH=n
MAX_NB_WORDS = 10000    # vocabulary中最多保留多少最高频的字
EMBEDDING_DIM = 800     # embedding层的维度以及第一层lstm输入维度
EMBEDDING_DIM_2 = 1600     # 第二层lstm的输出维度
BATCH_SIZE = 64    # batch的大小
EPOCHS = 50    # 迭代次数

def cut_words(file_name):
    """
    功能：将file中的内容返回按“字”分割的列表
    :param file_name: 文件名称
    :return: 文件内容按“字”分割的列表
    """
    with open(file_name, 'r', encoding='utf8') as f:
        content = f.read().replace('\n', '。')   # 使用句号作为句子的结束符
        f.close()
    return list(content)

def map_words(cut_word_list):
    """
    将训练文本中的“字”形成字典：word2index和index2word
    :param cut_word_list:文件内容按“字”分割的列表
    :return:word2index和index2word， 2个字典类型的结果，分别以字为key，index为value和以index为key，字为value。
    """
    vocabulary = sorted(list(set(cut_word_list)))
    word_to_index = dict((w, i+2) for i, w in enumerate(vocabulary))
    word_to_index["PAD"] = 0   # 填补
    word_to_index["UNK"] = 1   # unknown
    index_to_word = dict((index, word) for word, index in word_to_index.items())

    # 这里需要将2个文件存储，以便在生成歌词时使用
    word_to_index_json = json.dumps(word_to_index)
    index_to_word_json = json.dumps(index_to_word)
    with open('./word_to_index_word.txt', 'w', encoding='utf8') as w:
        w.write(word_to_index_json)
        w.close()
    with open('./index_to_word_word.txt', 'w', encoding='utf8') as w:
        w.write(index_to_word_json)
        w.close()
    # print("len of word_to_index::", len(word_to_index))
    # print("len of index_to_word::", len(index_to_word))
    return word_to_index, index_to_word

def generate_train_data(cut_word_list, word_to_index):
    """
    构造训练集，并处理成keras可以接受的输入格式。
    :param cut_word_list: 按“字”分割之后的list
    :param word_to_index: word2index的映射字典
    :return:X_train, X_val, y_train, y_val：训练集和验证集
    """
    # 生成训练数据
    X_data = []
    y_data = []
    data_index = []
    n_all_words = len(cut_word_list)
    for i in range(0, n_all_words - SEQ_LENGTH - 1):
        seq_x_y = cut_word_list[i: i+SEQ_LENGTH + 1]   # 最后一个词是y，seq_x_y表示前seq_length个词是训练集，最后一个字是训练数据对应的y
        index_x_y = [word_to_index[elem] for elem in seq_x_y]    # 获取seq_x_y对应的index组成的列表
        data_index.append(index_x_y)
    np.random.shuffle(data_index)
    for i in range(0, len(data_index)):
        X_data.append(data_index[i][:SEQ_LENGTH])
        y_data.append(data_index[i][SEQ_LENGTH])

    # 将X_data变换成需要输入的tensor模式，将y_data变成one-hot模式
    X = np.reshape(X_data, (len(X_data), SEQ_LENGTH))
    y = np_utils.to_categorical(y_data)

    # 训练集合验证集分割。
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=33)

    return X_train, X_val, y_train, y_val


def model_lstm(X_train, X_val, y_train, y_val, word_to_index):
    """
    训练模型并保存参数设置。
    :param X_train:训练集X
    :param X_val:验证集X
    :param y_train:训练集y
    :param y_val:验证集y
    :param word_to_index:word索引
    :return:history_record，这个主要为了后续画图获取loss使用（也可以不画图，直接使用tensorboard）
    """
    input_shape = (SEQ_LENGTH,)
    x_train_in = Input(input_shape, dtype='int32', name="x_train")

    # word_index存储的是所有vocabulary的映射关系
    nb_words = min(MAX_NB_WORDS, len(word_to_index))
    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, input_length=SEQ_LENGTH)(x_train_in)
    print("embedding layer is::", embedding_layer)
    print("build model.....")

    # return_sequences=True表示返回的是序列，否则下面的LSTM无法使用，但是如果下一层不是LSTM，则可以不写
    lstm_1 = LSTM(EMBEDDING_DIM, name="LSTM_1", return_sequences=True)(embedding_layer)
    lstm_2 = LSTM(EMBEDDING_DIM_2, name="LSTM_2")(lstm_1)
    dense = Dense(nb_words, activation="softmax", name="Dense_1")(lstm_2)

    model = Model(inputs=x_train_in, outputs=dense)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    print("Train....")

    # save tensorboard info
    tensorboard = TensorBoard(log_dir='./tensorboard_log/')
    # save best model.
    checkpoint = ModelCheckpoint(filepath='./model_epoch50_2lstm_1dense_seq50_phrase_based_best.h5',
                                 monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, period=1, verbose=1)
    callback_list = [tensorboard, checkpoint]

    history_record = model.fit(X_train,
                              y_train,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=(X_val, y_val),
                              callbacks=callback_list
                              )
    model.save('./model_epoch50_2lstm_1dense_seq50_phrase_based_best.h5')
    return history_record

def plot_accuray(history_record):
    """
    plot the accuracy and loss line. 若使用tensorboard，则可以不使用
    :param history_record:
    :return:
    """
    accuracy_train = history_record.history["acc"]
    accuracy_val= history_record.history["val_acc"]
    loss_train = history_record.history["loss"]
    loss_val = history_record.history["val_loss"]
    epochs = range(len(accuracy_train))
    plt.plot(epochs, accuracy_train, 'bo', label='Training accuracy')
    plt.plot(epochs, accuracy_val, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss_train, 'bo', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_name = "../train_data/all_5.txt"
    cut_word_list = cut_words(file_name)
    word_to_index, index_to_word = map_words(cut_word_list)
    X_train, X_val, y_train, y_val = generate_train_data(cut_word_list, word_to_index)
    history_record = model_lstm(X_train, X_val, y_train, y_val, word_to_index)
    # plot_accuray(history_record)



