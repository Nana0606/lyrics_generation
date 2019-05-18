# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/11/30 13:16

"""
爬取网易云音乐的歌词
Step1：获取歌手专辑id信息
Step2：根据专辑id获取这张专辑中包含的歌曲id
Step3：根据歌曲id爬取歌词
"""

import requests
import lxml.etree as etree
import os
import json
import re

def get_album_links(url, album_headers, f_path):
    """
    获取专辑名称和专辑id，将其存储到文件中，并调用get_lyrics_list()函数
    :param html:
    :return:
    """
    album_ids = []

    # 获取专辑名称数据
    response_albums = requests.get(url, headers=album_headers)
    pattern = re.compile(r'<div class="u-cover u-cover-alb3" title=(.*?)>')
    titles = re.findall(pattern, response_albums.text)

    # 判断是否文件已存在，若存在则删除
    if os.path.exists(f_path + "AlbumInfo.txt"):
        os.remove(f_path + "AlbumInfo.txt")
    if os.path.exists(f_path + "Lyrics.txt"):
        os.remove(f_path + "Lyrics.txt")

    # 获取专辑id并存储数据
    with open(f_path+"AlbumInfo.txt", 'a', encoding='utf8') as f:
        for title in titles:
            # 替换掉双引号，避免对正则化解析出现干扰
            title_handle = title.replace('\"', '')
            id_elem = re.compile(r'<a href="/album\?id=(.*?)" class="tit s-fc0">%s</a>' % title_handle)
            album_id = re.findall(id_elem, response_albums.text)   # 获取专辑id
            if len(album_id) == 1:
                f.write(title + "\t" + str(album_id[0]) + "\n")    # 追加写入文件
                album_ids.append(album_id[0])
            elif len(album_id) == 0:
                print("无对应的id")
            else:
                print("出错错误，一个专辑title对应多个id::", title)
    f.close()
    print("专辑爬取成功")
    return album_ids

def get_lyrics_list(album_ids, lyrics_list_url_current, lyrics_list_headers, f_path):
    """
    通过专辑的id获取没张专辑的歌曲及歌曲id
    :param album_links: 专辑ids
    :param lyricsList_url_row:
    :param lyrics_list_headers:
    :return:
    """
    with open(f_path + "lyricsList.txt", 'a', encoding='utf-8') as f:
        for album_id in album_ids:
            url = lyrics_list_url_current + str(album_id)
            print("url is::", url)
            response_lyrics_list = requests.get(url, headers=lyrics_list_headers)
            html_lyrics_list = etree.HTML(response_lyrics_list.text)
            lyric_list = html_lyrics_list.xpath('//ul[@class="f-hide"]//a')

            for lyric in lyric_list:
                html_data = str(lyric.xpath('string(.)'))
                # 获取歌曲的id
                pattern = re.compile(r'<a href="/song\?id=(\d+?)">%s</a>' % html_data)
                items = re.findall(pattern, response_lyrics_list.text)
                if len(items) == 1:
                    f.write(html_data + "\t" + str(items[0]) + "\n")
                elif len(items) == 0:
                    print("无歌曲id")
                else:
                    print("出现错误，一首歌曲的title对一个多个id::", html_data)
                print("歌曲::%s, 歌曲ID::%s 写入文件成功" % (html_data, items))
    f.close()

def get_lyrics(lyrics_headers, f_path):
    """
    通过歌曲id获取歌词
    :param lyrics_headers: 头文件
    :return:
    """
    # 直接读取所有内容
    with open(f_path + 'lyricsList.txt', 'r', encoding='utf8') as f:
        list_of_line = f.readlines()
    count = 1
    for elem in list_of_line:
        song_name = elem.split('\t')[0]
        song_id = elem.split('\t')[1]
        url = "http://music.163.com/api/song/lyric?" + "id=" + str(song_id) + '&lv=1&kv=1&tv=-1'
        response = requests.get(url, headers=lyrics_headers)
        json_content = json.loads(response.text)
        try:
            lyric = json_content['lrc']['lyric']
            pattern = re.compile(r'\[.*\]')
            lrc = str(re.sub(pattern, "", lyric).strip())
            with open(f_path + "歌曲名-" + song_name + ".txt", 'w', encoding='utf-8') as w:
                w.write(lrc)
                w.close()
            count += 1
        except:
            print("歌曲有错误，歌名为：%s。" % song_name)
    print("共爬取歌曲数量为：%s" % count)

if __name__ == '__main__':
    # 存储路径
    f_path = "./lyrics_yixun_chen/"
    if not os.path.exists(f_path):
        os.mkdir(f_path)

    # 专辑地址和headers
    singer_id = 2116   # 歌手id，可以在网易云音乐网站上搜索自己喜欢歌手的ID
    album_url = "https://music.163.com/artist/album?id=" + str(singer_id) + "&limit=100&offset=0"
    album_headers = {
        'Accept': 'ext/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Cookie': '_iuqxldmzr_=32; _ntes_nnid=b9e9c8a460bfdbc1250ffc5908f95cad,1543554353478; _ntes_nuid=b9e9c8a460bfdbc1250ffc5908f95cad; __utma=94650624.345999225.1543554354.1543554354.1543554354.1; __utmc=94650624; __utmz=94650624.1543554354.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; WM_TID=N3LQ47ihEYOKTieS18tLGKdN8q6R0iyt; WM_NI=VJ7qPEylWXfPBYYl3aMJisRzvjArZ%2BpJ9ES13N1zv1m4f9VnfiHOdDEdcZWOUY6xS9Gi27GgC4pLKvWki7aHQISPLz9y0Uo3kGEPO5874RygU9DtXL3P1LY1%2BiVxtw77UEk%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6ee8bd05d8a96ba85ef7af2bc8fa6c14f929b9e85f25cbbb5afa2d874bc96a2b8d12af0fea7c3b92a9696f992d34fbbb499b9e55c8287978db26b8f96a794d172b6b68f93d921b496f7a6d23bf192ff9bf53bafecaadad566a6b788b0e75087bcadbbd6459787fa99d23385b7fed8c66d81ecc0adca7e95b5a7d8e44493979db1cf72fba700b6ee39a9998bb1ce3ffce9fcb3fc6d97e78c95db48f190ffaddc3be9b79a99f77df5e99ba6ea37e2a3; JSESSIONID-WYYY=pNCF%2B8xzB2jTGWW7r7JlavTaS0YVMSZBP9THDnXZp86OQ3Aqo5WpW6dr3h6FR3hgevYdmOdO8N7aubiagD%2FhBrf%2BYd%2BcXtBehyUotNH%2BCs%5CqZXKRbf4Pyt6fU1tl7UCsXBvbe6b5%2BQwZ%5Cuth8Shm4fRFdkApHsDIEA9tuUYQYDB7BYuo%3A1543559635032; __utmb=94650624.51.10.1543554354',
        'Host': 'music.163.com',
        'Referer': 'https://music.163.com/',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'
    }
    # 获取所有专辑的地址
    album_ids = get_album_links(album_url, album_headers, f_path)

    # 专辑主页，专辑详情页面的headers和专辑list页面的headers一样
    lyrics_list_url_current = "http://music.163.com/album?id="
    get_lyrics_list(album_ids, lyrics_list_url_current, album_headers, f_path)

    # 获取歌词的API的headers
    lyrics_headers = {
        'Request URL': 'http://music.163.com/weapi/song/lyric?csrf_token=',
        'Request Method': 'POST',
        'Status Code': '200 OK',
        'Remote Address': '59.111.160.195:80',
        'Referrer Policy': 'no-referrer-when-downgrade'
    }
    get_lyrics(lyrics_headers, f_path)