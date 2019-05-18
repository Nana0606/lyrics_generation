# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/12/5 21:32

"""
功能：使用train_seq2seq_word_based.py保存的模型和参数生成歌词。
"""

from keras.models import load_model, Model
from keras.layers import Input
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

LATENT_DIM = 1000    # encoder结果语义向量的维度，需要和train文件中的配置一样


def load_param(model_file, word2index_input_file, index2word_input_file, word2index_target_file, index2word_target_file):
    """
    加载模型和参数
    :param model_file: 模型和参数路径
    :param word2index_input_file: 输入的word2index
    :param index2word_input_file:  输入的index2word
    :param word2index_target_file:   输出的word2index
    :param index2word_target_file:   输出的index2word
    :return:
    model：训练之后的模型和参数
    word2index_input, index2word_input_new： 输入的index2word和word2index
    word2index_target, index2word_target_new： 输出的index2word和word2index。
    """
    # get model.
    model = load_model(model_file)
    # get the word2index and index2word data.
    with open(word2index_input_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        word2index_input = json.loads(json_obj)
        f.close()
    with open(index2word_input_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        index2word_input = json.loads(json_obj)
        f.close()
    index2word_input_new = {}
    for key, value in index2word_input.items():
        index2word_input_new[int(key)] = value

    with open(word2index_target_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        word2index_target = json.loads(json_obj)
        f.close()
    with open(index2word_target_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        index2word_target = json.loads(json_obj)
        f.close()
    index2word_target_new = {}
    for key, value in index2word_target.items():
        index2word_target_new[int(key)] = value
    print("word2index is：：", index2word_input_new)
    print("word2index is：：", word2index_input)
    print("index2word is::", index2word_target_new)
    return model, word2index_input, index2word_input_new, word2index_target, index2word_target_new

def get_model(model):
    """
    通过已经加载的模型，获取模型中的encoder和decoder
    :param model: 模型和参数路径
    :return: encoder_model, decoder_model: 编码模型和解码编码，编码模型用于对上句编码，解码模型用于生成下句
    """
    encoder_inputs = model.get_layer(name="encoder_inputs").input
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name="encoder_outputs").output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.get_layer(name="decoder_inputs").input
    decoder_state_input_h = Input(shape=(LATENT_DIM,), name='input_3')
    decoder_state_input_c = Input(shape=(LATENT_DIM,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h_dec, state_c_dec = model.get_layer(name="decoder_LSTM")(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = model.get_layer(name="Dense_1")(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] +decoder_states)
    return encoder_model, decoder_model


def sample(preds, diversity = 1.0):
    """
    得到最大概率的phrase对应的下标
    :param preds:
    :param diversity:
    :return:
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def decode_sequence(encoder_model, decoder_model, input_seq_rep, word2index_target, index2word_target):
    """
    生成句子
    :param encoder_model:
    :param decoder_model:
    :param input_seq_rep: 输入的句子
    :param word2index_target: 输出的word2index
    :param index2word_target: 输出的index2word
    :return: decoded_sentence_all， generated sentence
    """
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq_rep)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(word2index_target)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word2index_target['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).

    decoded_sentence_all = ''
    max_decoder_seq_length = 20   # 生成的每个句子的最大长度
    count = 0
    while count < 10:   # 生成10个句子
        stop_condition = False    # 到达行结束符
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token1, :])
            sampled_token_index = sample(output_tokens[0, -1, :], 1)
            sampled_char = index2word_target[sampled_token_index]    # 生成的新的字
            decoded_sentence += sampled_char

            if(sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
                if sampled_char != '\n':
                    print("长度为20，直接结束，不用等待生成换行符！")
                    decoded_sentence += '\n'   # 若不自动换行，则手动添加一个结束符
                stop_condition = True

            if decoded_sentence != '\n':   # 防止出现没有内容的行
                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, len(word2index_target)))
                target_seq[0, 0, sampled_token_index] = 1.
                # Update states
                states_value = [h, c]
        if decoded_sentence != '\n':   # 防止出现没有内容的行
            count += 1
            decoded_sentence_all += decoded_sentence
    return decoded_sentence_all

def get_seq_representation(input_seq, word2index_input):
    """
    生成输入句子的表示
    :param input_seq: 输入的句子
    :param word2index_input:  输入的word2index
    :return:seq_rep， 输入的one-hot表示
    """
    seq_rep = np.zeros((1, 16, len(word2index_input)), dtype='float32')
    cut_word_list = list(input_seq)
    print("cut_word_list::", cut_word_list)
    for i in range(0, len(cut_word_list)):
        seq_rep[0, i, int(word2index_input.get(cut_word_list[i], 1))] = 1   # "1"表示"<UNK>"
    return seq_rep


if __name__ == '__main__':
    model_file = './model_seq2seq_100epoch_best.h5'
    word2index_input_file = './word_to_index_seq2seq_input.txt'
    index2word_input_file = './index_to_word_seq2seq_input.txt'
    word2index_target_file = './word_to_index_seq2seq_target.txt'
    index2word_target_file = './index_to_word_seq2seq_target.txt'
    model, word2index_input, index2word_input_new, word2index_target, index2word_target_new = load_param(model_file, word2index_input_file, index2word_input_file, word2index_target_file, index2word_target_file)
    encoder_model, decoder_model = get_model(model)
    input_seq = "喧嚣的人群"
    input_seq_rep = get_seq_representation(input_seq, word2index_input)
    decoded_sentence = decode_sequence(encoder_model, decoder_model, input_seq_rep, word2index_target, index2word_target_new)
    print(decoded_sentence)
