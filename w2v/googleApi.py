import requests
import csv
import threading, time
from bs4 import BeautifulSoup
import nltk

_BLACK = '\u001b[30m'
_RED = '\u001b[31m'
_GREEN = '\u001b[32m'
_YELLOW = '\u001b[33m'
_BLUE = '\u001b[34m'
_MAGENTA = '\u001b[35m'
_CYAN = '\u001b[36m'
_WHITE = '\u001b[37m'
_END = '\u001b[0m'

_Bold = '\u001b[1m'
_Underline = '\u001b[4m'
_Reversed = '\u001b[7m'


def END_():
    print(_END)


def GREEN_():
    print(_GREEN)


def BLUE(text: str):
    return _BLUE + text + _END


def BLACK(text: str):
    return _BLACK + text + _END


def RED(text: str):
    return _RED + text + _END


def GREEN(text: str):
    return _GREEN + text + _END


def YELLOW(text: str):
    return _YELLOW + text + _END


def MAGENTA(text: str):
    return _MAGENTA + text + _END


def CYAN(text: str):
    return _CYAN + text + _END


def WHITE(text: str):
    return _WHITE + text + _END


def Bold(text: str):
    return _Bold + text + _END


def Underline(text: str):
    return _Underline + text + _END


def Reversed(text: str):
    return _Reversed + text + _END


def reform_dev(str_data: str):
   list_data = str_data.replace("\xa0...", "").replace("\n", " ")\
       .replace(
       "Wikipedia", "").replace("synonym", "")
   new_str = " "
   list_data = nltk.word_tokenize(list_data)
   for word in list_data:
       if word in [".", "!", "?"]:
           with open("w10.txt", "a", encoding='utf-8') as data:
               data.write(new_str+word+"\n")
               new_str = ""
       else:
           if len(word) >= 2:
               if "//" in word:
                   pass
               else:
                   new_str += word + " "


def google_dev(q):
   try:
       s = requests.Session()
       url = f'https://de.wikipedia.org/wiki/{q}'
       res = s.get(url)
       # print(res.text)
       soup = BeautifulSoup(res.text, "html.parser")
       reform_dev(soup.text)
   except:
       pass
   for a in soup.find_all('a', href=True):
       link = a['href']
       if "wiki" in link:
           url = f'https://de.wikipedia.org{link}'
           print("Found the URL:", url)
           try:
               res = s.get(url)
               soup1 = BeautifulSoup(res.text, "html.parser")
               reform_dev(soup1.text)
           except:
               pass


# def quarry(word: str):
#    google_dev(word)


# def loader():
#    global end
#    with open(f"wls.txt", "r", encoding='utf-8') as data:
#        for line in data.readlines():
#            line = line.strip()
#            end.append(line)

#    print(len(end))

# https://deutsch.lingolia.com/de/leseverstehen


# def reform_dev(str_data: str):
#    global allln
#    list_data = str_data.replace("\xa0...", "").replace("\n", " ").replace("/", " ") \
#        .replace(
#        "Duden", "")
#    new_str = " "
#    list_data = nltk.word_tokenize(list_data)
#    for word in list_data:
#        if word in [".", "!", "?"]:
#            allln += 1
#            data.write(new_str + word + "\n")
#            new_str = ""
#        else:
#            if len(word) >= 2:
#                if "//" in word:
#                    pass
#                else:
#                    new_str += word + " "


# def google_dev(q):
#    global tnum
#    try:
#        s = requests.Session()
#        url = q
#        res = s.get(url)
#        soup = BeautifulSoup(res.text, "html.parser")
#        reform_dev(soup.text)
#        for a in soup.find_all('a', href=True):
#            url = a['href']
#            if url not in list_herf:
#                # print(BLUE("Found new URL:" + url))
#                list_herf.append(url)
#                url = "https://de.wikipedia.org" + url
#                if tnum >= 1500:
#                    quarry(url)
#                else:
#                    threading.Thread(target=quarry, args=[url]).start()
#                    tnum += 1
#                pass
#    except:
#        pass


# def quarry(url):
#    # print(api.load(url, return_path=True))
#    google_dev(url)


import gensim.downloader as api
import pydeepl

# def loader():
#    global end
#    with open(f"wls.txt", "r", encoding='utf-8') as data:
#        for line in data.readlines():
#            line = line.strip()
#            end.append(line)
#    print(len(end))

# https://deutsch.lingolia.com/de/leseverstehen
# tnum = 0
# list_herf = []
# allln = 0
# l = """
# https://de.wikipedia.org/wiki/Leben
# https://de.wikipedia.org/wiki/Universum
# https://de.wikipedia.org/wiki/Ich
# https://de.wikipedia.org/wiki/Literatur
# https://de.wikipedia.org/wiki/Reden
# https://de.wikipedia.org/wiki/Wissenschaft
# https://de.wikipedia.org/wiki/Geschichte
# https://de.wikipedia.org/wiki/Naturwissenschaft
# https://de.wikipedia.org/wiki/Biologie
# https://de.wikipedia.org/wiki/Geographie
# https://de.wikipedia.org/wiki/Gesundheitssystem
# https://de.wikipedia.org/wiki/Deutschland
# https://de.wikipedia.org/wiki/Kriminalität
# https://de.wikipedia.org/wiki/Ökologie
# https://de.wikipedia.org/wiki/Weltkrieg
# https://de.wikipedia.org/wiki/Geist
# https://de.wikipedia.org/wiki/Seele
# https://de.wikipedia.org/wiki/Krieg
# https://de.wikipedia.org/wiki/Neurologie
# https://de.wikipedia.org/wiki/World_Wide_Web
# https://de.wikipedia.org/wiki/Mathematik
# https://de.wikipedia.org/wiki/Psychologie
# https://de.wikipedia.org/wiki/Finanzen
# https://de.wikipedia.org/wiki/Land_(Deutschland)
# https://de.wikipedia.org/wiki/Bruttoinlandsprodukt
# https://de.wikipedia.org/wiki/Deutsche_Sprache
# https://de.wikipedia.org/wiki/Ethik
# https://de.wikipedia.org/wiki/Moral
# https://de.wikipedia.org/wiki/Weltkrieg
# https://de.wikipedia.org/wiki/Krieg
# https://de.wikipedia.org/wiki/Fußball
# https://de.wikipedia.org/wiki/Politik
# """.split("\n")

# tok = '[ " @ : # % \' ( ) * +  -  / : ; < = >  @ \ [ \ ] ^ _ ` { | } ~ 1 2 3 4 5 6 7 8 9 0 ’ ” “ ′ ‘ \\ \ ]'.split(" ")
# a = open("isaaT06.txt", "r", encoding='utf-8')
# with open("isaaT06E.txt", "a", encoding='utf-8') as data:
#    for data_s in a.readlines():
#        for f in tok:
#            data_s = data_s.replace(f, "")
#        data.write(data_s)
# for i in l:
#
#    threading.Thread(target=quarry,
#                     args=[i]).start()
#
##quarry("20-newsgroups") #C:\Users\hausm/gensim-data\text8\text8.gz
##threading.Thread(target=quarry,
##                 args=["text8"]).start()
##
##threading.Thread(target=quarry,
##                 args=["wiki-en"]).start()
##
##threading.Thread(target=quarry,
##                 args=["wiki-de"]).start()
##
##threading.Thread(target=quarry,
#                 #args=["quora-duplicate-questions"]).start()
# while True:
#    if input(": ") == "x":
#        break
#    if input(": ") == "":
#        print(GREEN("------------------------------"))
#        print(GREEN("|----------------------------|"))
#        print(YELLOW("|-Threads---list_herf--sätze-|"))
#        print(CYAN(f"|--{tnum}----{len(list_herf)}----{allln}--|"))
#        print(GREEN("|----------------------------|"))
#        print(GREEN("------------------------------"))
##
# with open("links1.txt", "a", encoding='utf-8') as data:
#    for line in list_herf:
#        data.write(line+"\n")
#    data.close()
#


import nltk
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field
import gensim
from gensim import models
import speech_recognition as sr
import time
from gtts import gTTS
import os
from playsound import playsound


def test(a_str):
    a_str = a_str.lower().replace("ä", "A").replace("ö", "O").replace("ü", "U")
    a_str = a_str.split(" ")
    end = ""
    for i in a_str:
        if i.isascii():
            end += i.replace("A", "ä").replace("O", "ö").replace("U", "ü") + " "
    return end


dc = {"s": [], "e": []}
corpus = []
import re
import time

for i in range(1, 382):
    print("||", i)
    data = ""
    with open(f"all\\t1 ({i}).xml", 'r', encoding="ISO-8859-1") as f:
        for line in f.readlines():
            line = line.strip()
            data += line + "\n"

    data = data.split("creatorList")
    creatorList = data[1].replace("\t", "")
    flip = True
    flop = False
    if True:
        for msb in data[2].split("\n"):
            if flop:
                flop = False
                if ">" in msb or "<" in msb or "\\" in msb or "/" in msb or "room" in msb or "nicknamestudentnickname" in msb or "nicknamestudentnickname" in msb:
                    pass
                else:
                    msb = msb.lower()
                    cleanString = re.sub('\W+', '', msb.replace(" ", "A")).replace("A", " ")
                    msb = re.sub('\d+', '', cleanString)
                    msb = test(msb)
                    corpus.append(msb)
                    if flip:
                        dc["s"].append(msb + "\n")
                        flip = False
                    else:
                        dc["e"].append(msb + "\n")
                        flip = True
            if msb == "<messageBody>":
                flop = True

# with open(f"Isaa\w.txt", 'r', encoding="utf-8") as f:
#    for line in f.readlines():
#        line = line.strip()
#        corpus.append(test(line.lower()))
#
# with open(f"Isaa\\a", 'r', encoding="utf-8") as f:
#    for line in f.readlines():
#        line = line.strip()
#        corpus.append(test(line.lower()))
#
# with open(f"isaaBA\w10.txt", 'r', encoding="utf-8") as f:
#    for line in f.readlines():
#        line = line.strip()
#        corpus.append(test(line.lower()))


# end = []
# for x in corpus:
#    cleanString = re.sub('\W+', '', x.replace(" ", "A")).replace("A", " ")
#    cleanString = re.sub('\d+', '', cleanString)
#    end += [cleanString]
# corpus = end
print(type(corpus))

# corpus += ["< > |"]
# corpus += [". ? ! , - _"]
# corpus += ["1 eins 2 zwei 3 drei 4 vier 5 fünf 6 sechs 7 seiben 8 acht 9 neun 0 null 10 zehn 1 2 3 4 5 6 7 8 9 0"]
#
# with open(f"IsaaB32\w1.txt", 'w', encoding="utf-8") as f:
#  for line in corpus:
#      f.write(line+"\n")
#
with open(f"IsaaB32\data3.txt", 'w', encoding="utf-8") as f:
    f.write(str(dc))

# tok_corp = [nltk.word_tokenize(sent) for sent in corpus]
#
# with open(f"IsaaB32\ln.txt", 'w', encoding="utf-8") as f:
#    f.write(str(len(tok_corp)))
#
# print(len(tok_corp))
#
##train the skip-gram model; default window=5
# model = models.word2vec.Word2Vec(tok_corp, size=32, min_count=1, workers=124)
# model.save('isaaB322')


x = """  
          45 , 53 , 6 ,19 -> | 123 , 41
        - 95 , 7 , 105 -> | 207 
        - 42 , 45 , 22 -> | 109
        - 34 , 33 , 8 , 7 -> | 82
        - (3) = 
        - 9, 8, 103, 32 ->
        - (1) =
        - (14.50)
        
        """
