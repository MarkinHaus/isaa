import pandas as pd
import nltk
import gensim
from gensim import models
from gensim import downloader
import os

import threading
# with open("newsgroups.txt", "a", encoding='utf-8') as data:
#    for i in document:
#        data.write(i+"\n")
#
#    for i in corpus.get_texts():
#        data.write(i+"\n")
#


from googletrans import Translator


def loade():
    files = """DataTest\\20-newsgroups\\20-newsgroups
    DataTest\\quora-duplicate-questions\\quora-duplicate-questions
    DataTest\\semeval-2016-2017-task3-subtaskA-unannotated\\semeval-2016-2017-task3-subtaskA-unannotated
    DataTest\\semeval-2016-2017-task3-subtaskBC\\semeval-2016-2017-task3-subtaskBC
    DataTest\\text8\\text8
    """.split("\n")
    files = ["DataTest\\quora-duplicate-questions\\quora-duplicate-questions"]
    tok = '[ " # % \' ( ) * +  -  / : ; < = >  @ \ [ \ ] ^ _ ` { | } ~ 1 2 3 4 5 6 7 8 9 0 ’ ” “ ′ ‘ \\ \ ]'.split(" ")

    def m(f_name):
        translator = Translator(service_urls=[
            'translate.google.com',
            'translate.google.co.kr',
        ])
        print(f_name)
        data = open(f_name, "r", encoding='utf-8')
        i = 0
        for line in data.readlines():
            if i == 100:
                i = 0
                e.flush()
            i += 1
            line = eval(line)
            info = line["data"]
            for data_s in info.split("\n"):
                if "@" in data_s or ":" in data_s:
                    pass
                else:
                    for f in tok:
                        data_s = data_s.replace(f, "")
                    # try:
                    if data_s:
                        aut = translator.translate(data_s, dest="de")
                        e.write(aut.text + "\n")
                    # except:
                    #    print("ERROR")
                    #    pass
        data.close()

    e = open("IsaaBE.txt", "a", encoding='utf-8')
    m("DataTest\\20-newsgroups\\20-newsgroups")
    e.close()


# D:\\programmieren\\project_py\\isaa\\w2v\\DataTest\\20-newsgroups\\20-newsgroups.gz

# https://www.google.com/search?safe=active&rlz=1C1CHBF_deDE849DE849&biw=1745&bih=842&sxsrf=ALeKk00zZQB_bXIxcYzTLaQMjFwFS1lV_Q%3A1596761787379&ei=u6YsX9DhFtL8kwWNgo-YCg&q=word2vec+dataset+download&oq=word2vec+dataset+&gs_lcp=CgZwc3ktYWIQARgBMgQIIxAnMgYIIxAnEBMyBggAEBYQHjIGCAAQFhAeMgYIABAWEB4yBggAEBYQHjIGCAAQFhAeMgYIABAWEB4yBggAEBYQHjIGCAAQFhAeOgQIABBHUO2SBVjtkgVgmZ0FaABwAXgAgAFwiAFwkgEDMC4xmAEAoAEBqgEHZ3dzLXdpesABAQ&sclient=psy-ab
# nltk.download('punkt')
#
# df = pd.read_csv('knowlege2.csv')

# x = df['definition'].values.tolist()
# y = df['synonym'].values.tolist()

# corpus = x + y
# corpus = []
#   ##
#
# for i in range(12):
#   with open(f"D:\\programmieren\\project_py\\isaa\\w2v\\isaaT1\\isaaT11 ({i + 1}).txt", 'r') as f:
#       for line in f.readlines():
#           line = line.strip()
#           corpus.append(line)
# print(len(corpus))
#
# corpus = corpus + corpus
# sentences = models.word2vec.Text8Corpus('text8')
# print(sentences.max_sentence_length, sentences)
## train the skip-gram model; default window=5
# model = models.word2vec.Word2Vec(sentences, size=32)
#
# model.save('isaaB1T3N32')

# model = models.KeyedVectors.load_word2vec_format('../wordvecs/de3E9.bin', binary=True)
# model.most_similar("leben")
# lebten wachsen wohnen lebenden Hindus beten fallen sterben gehen liegen


# print(len(corpus))
# tok_corp = [nltk.word_tokenize(sent) for sent in corpus]
# model = gensim.models.Word2Vec(tok_corp, min_count=1, size=32)
##
# model.save('isaaB1T2N32')
##
### nachschlagen.Synonym Sprache.Sein
# import time


import speech_recognition as sr
import time
from gtts import gTTS
import os
from playsound import playsound


def test():
    def pros(aout1, mod, text):
        t = time.time()
        aout = " "
        try:
            if len(text.split(" ")) == 2:
                m1 = mod.wv.get_vector(text.split(" ")[0])
                m2 = mod.wv.get_vector(text.split(" ")[1])

                m3 = m1 + m2

                end = mod.wv.similar_by_vector(m3)

                for i in end:
                    aout += i[0] + " "

            else:
                m = mod.wv.most_similar(text.split(" ")[0])
                for i in m:
                    aout += i[0] + " "
            output = gTTS(text=aout, lang=language)
            name = f"temp{time.time()}.mp3"
            print(text, aout1, aout, time.time() - t)
            output.save(name)
            playsound(name)
            os.remove(name)
        except:
            aout = "-"
            print(text, aout1, aout, time.time() - t)
            pass

    # isaa interaktive selflernig acistens a.i
    r = sr.Recognizer()

    language = 'de'
    import torch
    #model_isaaB321 = gensim.models.Word2Vec.load('isaaB321')
    model_isaaB322 = gensim.models.Word2Vec.load('isaaB322')
    #model_isaa32 = gensim.models.Word2Vec.load('isaa32')
    #model_isaaB = gensim.models.Word2Vec.load('isaaB')
#
    #print("model_isaa32 ", model_isaa32.wv.vector_size, len(model_isaa32.wv.vocab))
    #print("model_isaaB321 ", len(model_isaaB321.wv.vocab))
    #print("model_isaaB322 ", len(model_isaaB322.wv.vocab), " - ", len(model_isaaB322.wv.vocab.keys()))
    #print("model_isaaB ", model_isaaB.wv.vector_size, len(model_isaaB.wv.vocab))




    with sr.Microphone() as source:
        while True:
            try:
                print("w...")
                audio = r.listen(source, phrase_time_limit=4)
                text = r.recognize_google(audio, language="de_DE")
#
            except:
                text = "-----"
            text = text.replace("Martin", "Markin")
            #pros("model_isaaB    ", model_isaaB, text)
            #pros("model_isaa32   ", model_isaa32, text.lower())
            #pros("model_isaaB321 ", model_isaaB321, text.lower())
            pros("model_isaaB322 ", model_isaaB322, text.lower())


            if text == "Ende":
                break


test()


# --
#     --    O
#   ---   </\_
#  ---  -\/\
#    ---   /_
