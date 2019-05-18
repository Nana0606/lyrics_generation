# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/12/2 11:32
"""
主要是对歌词进行处理，将所有歌词拼接到一个文件中，并将以下无用数据去除：
1、空行
2、特殊字开头：
    ["编曲", "编辑", "歌名", "歌手", "专辑", "木吉他", "贝斯", "发行日", "曲", "监制", "制作人", "中乐演奏", "其余所有乐器演奏", "演奏", "和音",
                 "联合制作", "制作", "录音", "混音", "录音室", "混音室", "录音师", "混音师", "统筹", "制作统筹", "执行制作", "母带后期处理", "企划", "鼓",
                 "合声", "二胡", "乌克丽丽", "过带", "Bass", "Scratch", "OP", "Guitar", "SP", "Bass", "SCRATCH", "Programmer", "弦乐", "小提琴",
                 "女声", "Cello solo", "Piano", "吉他", "钢琴", "os", "弦乐", "和声", "DJ", "Tibet", "Violin", "Viola", "Cello", "和声", "母带",
                 "音乐", "打击乐", "Vocal", "次中音", "长号", "小号", "Music", "监制", "作词", "词/曲", "箫", "筝", "作词", "作曲", "Program", "键盘", "制作"]
3、将标点符号去除
"""
import os
import re

def handle_lyrics(f_path, del_words):
    """
    对歌词进行处理。注意为了在seq2seq模型中使用，所以在每一首歌词后面添加一个换行
    :param f_path: 歌词文件夹路径
    :param del_words: 需要去掉的内容
    :return:
    """
    files_name = os.listdir(f_path)
    print("len of files_name::", len(files_name))
    with open("./all_5.txt", 'a', encoding='utf8') as w:
        for file_name in files_name:
            flag_contain = 0    # 此文件是否包含歌词信息，若为空，则不需要最后添加空行
            with open(f_path + file_name, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    flag_del = 0    # 此行是否需要去除
                    for del_word in del_words:
                        if del_word in line:
                            flag_del = 1    # 若有“作词”等开头的句子，说明不是歌词，则直接将此行去掉
                            break
                    if flag_del == 1:
                            continue
                    space_except_char = """
                    """   # 爬取到的特殊的换行符，注意：此符号不是“\n”
                    # 若一句话中含有特殊符号，则使用特殊符号分割
                    line_splits = [elem for elem in re.split(r'[ \s+_+#-:：.\?"\(\),~.。、，…？「」（）{}+=\u3000\n]', re.sub("[A-Za-z0-9]", "", line.replace(space_except_char, '')).strip()) if elem != "" and elem !='\n' and elem != '\r\n']
                    for line_split in line_splits:
                        if line_split.strip() != '' and line_split.strip() !='\r\n':
                            flag_contain = 1
                            w.write(line_split.strip() + "\n")
            f.close()
            if flag_contain == 1:   # 每首歌最后添加一个空行
                w.write("\n")
    w.close()

if __name__ == '__main__':
    f_path = "./lyrics_zongsheng_li_handle/"   # dataCrawling.py爬取结果的路径。可以先处理下，比如将乱七八糟的句子去除等，所以这里文件名是处理过的爬取数据目录。
    del_words = ["编曲", "编辑", "歌名", "歌手", "专辑", "木吉他", "贝斯", "发行日", "曲", "监制", "制作人", "中乐演奏", "其余所有乐器演奏", "演奏", "和音",
                 "联合制作", "制作", "录音", "混音", "录音室", "混音室", "录音师", "混音师", "统筹", "制作统筹", "执行制作", "母带后期处理", "企划", "鼓",
                 "合声", "二胡", "乌克丽丽", "过带", "Bass", "Scratch", "OP", "Guitar", "SP", "Bass", "SCRATCH", "Programmer", "弦乐", "小提琴",
                 "女声", "Cello solo", "Piano", "吉他", "钢琴", "os", "弦乐", "和声", "DJ", "Tibet", "Violin", "Viola", "Cello", "和声", "母带",
                 "音乐", "打击乐", "Vocal", "次中音", "长号", "小号", "Music", "监制", "作词", "词/曲", "箫", "筝", "作词", "作曲", "Program", "键盘", "制作"]
    handle_lyrics(f_path, del_words)