# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/12/2 13:37

"""
本文主要使用把歌词生成当做序列模型来做，使用seq2seq模型，其中，每个cell使用的是lstm。
代码包括模型及参数的保存以及tensorboard-log的保存。
参考：https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Masking
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import os

# 指定GPU和最大占用的显存比例
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

SEQ_LENGTH = 20    # 限制每一个句子的最大长度
MAX_NB_WORDS = 10000    # vocabulary中最多保留多少最高频的字
LATENT_DIM = 1000    # encoder产生的语义向量维度
BATCH_SIZE = 64    # batch大小
EPOCHS = 10   # 迭代次数

def cut_words(file_name):
    """
    对中文文件进行分词，但是为了方便构造训练集（训练集需要知道哪个是句子的结尾），所以引用了句号。
    :param file_name: 文件内容的路径
    :return: cut_word_list, word_list：cut_word_list用于存储所有歌词，其每个元素是一个list，存储一首歌（一首歌使用的又是list of list结构，一个list是一句歌词）；
             （其实可以不用这么复杂，直接存储所有歌词，每句话使用一个list即可，但是这里之所有把每首歌封装起来，是为了生成训练集时不会出现X时第一首歌的句子，y是第二首歌的句子这话情况）
             word_list用于存储出现的所有字（不去重）。
    """
    word_list = []  # 用于存储所有的词组
    content = ""
    cut_word_list = []
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines() # 使用句号作为句子的结束符
        for line in lines:
            if line != '\n':    # 只有一首歌的最后才有空行
                content += line.replace("\n", "。")   # 将换行符替换为句号，作为句子的结束，因为为了方便表示，下面使用了句号作为每一句的分割
        f.close()
    songs = [song for song in content.split('\n') if song != '']
    for song in songs:
        current_song = []  # 用于存储当前歌词分词结果
        cut_res = list(song)  # 按“字”分割
        current_line = []
        for elem in cut_res:
            if elem != '。':    # 说明不是一句话的最后一个字
                current_line.append(elem)
            else:
                current_song.append(current_line)      # 一句话已经结束
                current_line = []
            word_list.append(elem)
        cut_word_list.append(current_song)
    return cut_word_list, word_list

def map_words(word_list, file_name):
    """
    生成word-to-index和index-to-word
    :return:word_to_index, index_to_word，都是字段格式，分别以字为key，index为value和以index为key，字为value
    """
    vocabulary = sorted(list(set(word_list)))
    word_to_index = dict((w, i+2) for i, w in enumerate(vocabulary))
    # 0下标用于masking，即如果下标为0，则 表示此时间步不计算在内
    word_to_index["PAD"] = 0    # 补0操作
    word_to_index["UNK"] = 1    # 表示未出现在vocabulary中的词语
    index_to_word = dict((index, word) for word, index in word_to_index.items())

    # 这里需要将2个文件存储，以便在生成歌词时使用
    word_to_index_json = json.dumps(word_to_index)
    index_to_word_json = json.dumps(index_to_word)
    with open('./word_to_index_seq2seq'+file_name+'.txt', 'w', encoding='utf8') as w:
        w.write(word_to_index_json)
        w.close()
    with open('./index_to_word_seq2seq'+file_name+'.txt', 'w', encoding='utf8') as w:
        w.write(index_to_word_json)
        w.close()
    return word_to_index, index_to_word

def generate_input_target_text(cut_word_list):
    """
    生成对应的训练集X和y，这里X和y都是分词过的词语list
    :param cut_word_list: 存储所有歌词，其每个元素是一个list，存储一首歌（一首歌使用的又是list of list结构，一个list是一句歌词）；
    :return:
    input_texts: 数据集X对应的list
    target_texts: 数据集y对应的list
    max_seq_input: X中样本的最大长度
    max_seq_target: y中样本的最大长度
    input_words: 数据集X中的字列表，用于统计词频，产生word2index
    target_words: 数据集y中的字列表，用于统计词频，产生index2word
    """
    # 生成X和y
    input_texts = []    # 数据集X
    target_texts = []    # 数据集y
    input_words = []    # 数据集X中的字列表，用于统计词频，产生word2index
    target_words = []    # 数据集y中的字列表，用于统计词频，产生index2word
    num_songs = len(cut_word_list)
    max_seq_input = 0   # 输入序列中的最大长度（模型中需要使用）
    max_seq_target = 0  # 输出序列中的最大长度（模型中需要使用）
    for i in range(0, num_songs):
        num_lines_eachSong = len(cut_word_list[i])
        for j in range(0, num_lines_eachSong - 1):
            input_texts.append(cut_word_list[i][j])
            input_words += (cut_word_list[i][j])
            max_seq_input = max(max_seq_input, len(cut_word_list[i][j]))
            # 开始和结束的位置分别是：“\t”和"\n"
            target_texts.append(["\t"] + cut_word_list[i][j + 1]+ ["\n"])
            target_words += (["\t"] + cut_word_list[i][j + 1] + ["\n"])
            max_seq_target = max(max_seq_target, len(["\t"] + cut_word_list[i][j + 1] + ["\n"]))
    return input_texts, target_texts, max_seq_input, max_seq_target, input_words, target_words

def generate_train_data(input_texts, target_texts, word_to_index_input, word_to_index_target, max_seq_input, max_seq_target):
    """
    构造训练集，并处理成keras可以接受的输入
    :param input_texts: 数据集X对应的list
    :param target_texts: 数据集y对应的list
    :param word_to_index_input: 以字为key，index为value。
    :param word_to_index_target: 以index为key，字为value.
    :param max_seq_input: X中样本的最大长度
    :param max_seq_target: y中样本的最大长度
    :return:
    np.array(encoder_input_data_train)，np.array(decoder_input_data_train)，np.array(decoder_target_data_train)：训练数据
    np.array(encoder_input_data_val)，np.array(decoder_input_data_val)，np.array(decoder_target_data_val)：验证数据
    """
    encoder_input_data = np.zeros((len(input_texts), max_seq_input, len(word_to_index_input)), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_seq_target, len(word_to_index_target)), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_seq_target, len(word_to_index_target)), dtype='float32')
    # 都是用one-hot方法表示，这里注意decoder_input_data和decoder_target_data，比如输入是："我喜欢动物"，则输出"喜欢动物<end>"
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, elem in enumerate(input_text):
            encoder_input_data[i, t, word_to_index_input[elem]] = 1
        for t, elem in enumerate(target_text):
            decoder_input_data[i, t, word_to_index_target[elem]] = 1
            if t>0:
                decoder_target_data[i, t-1, word_to_index_target[elem]] = 1

    # 分割测试集和训练集
    index = np.arange(len(input_texts))
    train_index, val_index = train_test_split(index, test_size=0.2, random_state=33)
    encoder_input_data_train = []
    decoder_input_data_train = []
    decoder_target_data_train = []
    encoder_input_data_val = []
    decoder_input_data_val = []
    decoder_target_data_val = []
    for idx in train_index:
        encoder_input_data_train.append(encoder_input_data[idx])
        decoder_input_data_train.append(decoder_input_data[idx])
        decoder_target_data_train.append(decoder_target_data[idx])
    for idx in val_index:
        encoder_input_data_val.append(encoder_input_data[idx])
        decoder_input_data_val.append(decoder_input_data[idx])
        decoder_target_data_val.append(decoder_target_data[idx])

    return np.array(encoder_input_data_train), np.array(decoder_input_data_train), np.array(decoder_target_data_train), np.array(encoder_input_data_val), np.array(decoder_input_data_val), np.array(decoder_target_data_val)


def model_lstm(word_to_index_input, word_to_index_target, encoder_input_data_train, decoder_input_data_train, decoder_target_data_train, encoder_input_data_val, decoder_input_data_val, decoder_target_data_val):
    """
    训练模型并保存参数设置。
    :param word_to_index_input: 输入内容word2index
    :param word_to_index_target: 输出内容word2index
    :param encoder_input_data_train，decoder_input_data_train，decoder_target_data_train:训练数据
    :param encoder_input_data_val，decoder_input_data_val，decoder_target_data_val:验证数据
    :return: history_record，这个主要为了后续画图获取loss使用（也可以不画图，直接使用tensorboard）
    """
    encoder_inputs = Input(shape=(None, len(word_to_index_input)), dtype='float32', name="encoder_inputs")
    encoder_inputs_masking = Masking(mask_value=0)(encoder_inputs)
    print("build model.....")

    # return_sequences=True表示返回的是序列，否则下面的LSTM无法使用，但是如果下一层不是LSTM，则可以不写
    encoder = LSTM(LATENT_DIM, name="encoder_outputs", return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs_masking)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, len(word_to_index_target)), dtype='float32', name="decoder_inputs")
    decoder_inputs_masking = Masking(mask_value=0)(decoder_inputs)   # 因为输入是一致长度，这个用于处理seq2seq的变长问题
    decoder_LSTM = LSTM(LATENT_DIM, name="decoder_LSTM", return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs_masking, initial_state=encoder_states)
    decoder_dense = Dense(len(word_to_index_target), activation='softmax', name="Dense_1")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    print("Train....")
    # save tensorboard info
    tensorboard = TensorBoard(log_dir='./tensorboard_log/')
    # save best model.
    checkpoint = ModelCheckpoint(filepath='./model_seq2seq_100epoch_best.h5',
                                 monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False,
                                 period=1,
                                 verbose=1)
    callback_list = [tensorboard, checkpoint]

    history_record = model.fit([encoder_input_data_train, decoder_input_data_train],
                               decoder_target_data_train,
                               batch_size=BATCH_SIZE,
                               epochs=EPOCHS,
                               validation_data=(
                               [encoder_input_data_val, decoder_input_data_val], decoder_target_data_val),
                               # validation_split=0.2
                               callbacks=callback_list
                               )
    model.save('./model_seq2seq_100epoch_final.h5')
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
    cut_word_list, word_list = cut_words(file_name)
    input_texts, target_texts, max_seq_input, max_seq_target, input_words, target_words = generate_input_target_text(cut_word_list)
    word_to_index_input, index_to_word_input = map_words(input_words, '_input')
    word_to_index_target, index_to_word_target = map_words(target_words, '_target')
    encoder_input_data_train, decoder_input_data_train, decoder_target_data_train, encoder_input_data_val, decoder_input_data_val, decoder_target_data_val = generate_train_data(input_texts, target_texts, word_to_index_input, word_to_index_target, max_seq_input, max_seq_target)
    history_record = model_lstm(word_to_index_input, word_to_index_target, encoder_input_data_train, decoder_input_data_train, decoder_target_data_train, encoder_input_data_val, decoder_input_data_val, decoder_target_data_val)
    # plot_accuray(history_record)


