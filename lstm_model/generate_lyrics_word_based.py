# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/12/2 15:14

"""
功能：使用train_lstm_word_based.py保存的模型和参数生成歌词。
"""

from keras.models import load_model
import numpy as np
import json

def load_param(model_file, word2index_file, index2word_file):
    """
    load model and word2index_file, index2word_file
    :param model_file:
    :param word2index_file:
    :param index2word_file:
    :return:
    """
    # get model.
    model = load_model(model_file)
    # get the word2index and index2word data.
    with open(word2index_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        word2index = json.loads(json_obj)
        f.close()
    with open(index2word_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        index2word = json.loads(json_obj)
        f.close()
    index2word_new = {}
    for key, value in index2word.items():
        index2word_new[int(key)] = value
    return model, word2index, index2word_new

def sample(preds, diversity = 1.0):
    """
    get the max probability index.
    :param preds: 预测结果
    :param diversity:
    :return:
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(start, model, word2index, index2word, SEQ_LENGTH, generate_maxlen):
    """
    generate lyrics according start sentence.
    :param start: startWith sentence
    :param model:
    :param word2index:
    :param index2word:
    :param maxlen: the length of generating sentence.
    :return:
    """
    sentence = start[:SEQ_LENGTH]   # 限制最开始的长度，sentence用于存储
    diversity = 1.0
    while len(sentence) < generate_maxlen:
        # 将最开始的句子分词，并存储到x_pred中
        x_pred = np.zeros((1, SEQ_LENGTH))    # 使用PAD填充

        min_index = max(0, len(sentence) - SEQ_LENGTH)    # 因为要通过前10个字，预测后一个字，获取前10个字的index
        for idx in range(min_index, len(sentence)):
            x_pred[0, SEQ_LENGTH - len(sentence) + idx] = word2index.get(sentence[idx], 1)   # '<UNK>' is 1

        preds = model.predict(x_pred, verbose=0)[0]   # 预测的概率
        next_index = sample(preds, diversity)   # 根据预测的概率采样确定下一个字
        next_word = index2word[next_index]
        if not (next_word == '。' and sentence[-1] == '。'):   # 防止出现一句话没有内容，只有句号
            sentence = sentence + next_word   # 每次都往后取一个
    return sentence


if __name__ == '__main__':
    model_file = './model_epoch50_2lstm_1dense_seq10_word_based_best.h5'
    word2index_file = './word_to_index_word.txt'
    index2word_file = './index_to_word_word.txt'
    model, word2index, index2word = load_param(model_file, word2index_file, index2word_file)
    start = "痴情"
    generate_maxlen = 200
    SEQ_LENGTH = 10
    sentence = generate(start, model, word2index, index2word, SEQ_LENGTH, generate_maxlen)
    print(sentence.replace("。", '\n'))  # 显示的时候使用换行更可观
