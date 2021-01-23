#  from BOT.unix.utils import *
from BOT.unix.model import *
from BOT.unix import *
from BOT.unix.get_corpus_y import YouTubeDataSetCaptions

print(version)

url = ["https://www.youtube.com/watch?v=mojirClzQEs", "https://www.youtube.com/watch?v=gnrQXMYJRRA",
       "https://www.youtube.com/watch?v=ht8cZrUQXB4", "https://www.youtube.com/watch?v=HrSXHs3LMlU",
       "https://www.youtube.com/watch?v=2IJvxeyjNGo", "https://www.youtube.com/watch?v=W4XkLx8sQgs",
       "https://www.youtube.com/watch?v=Gewb3-DUlJs", "https://www.youtube.com/watch?v=6avJHaC3C2U",
       "https://www.youtube.com/watch?v=NwPDJz-GIdY&t", "https://www.youtube.com/watch?v=ck4RGeoHFko",
       "https://www.youtube.com/watch?v=HAnw168huqA"]

name = "Issa_Z1"

model_data = ModelsData(name)
youtube_data_set_captions = YouTubeDataSetCaptions(name)
youtube_data_set_captions.load("uni_Z1")

# youtube_data_set_captions.append_conv(url, "de")
# youtube_data_set_captions.format()
# youtube_data_set_captions.crate_voc()
# youtube_data_set_captions.save("ys_test")

# train_data = open("..\\Transvormer\\dataen\\tarin\\conf.txt", "r", encoding="utf-8").read()
# train_data = train_data.replace("\\n", "")
# train_data = eval(train_data)
# out = []
# scr_list = train_data["s"]
# tar_list = train_data["e"]
#
# if len(scr_list) == len(tar_list):
#     l0 = len(scr_list)
#     for i in range(l0):
#         print(f"end {i / len(scr_list) * 100:.2f}%")
#         out.append([scr_list[i], tar_list[i]])
#     print(f"len {l0=} ")
# else:
#     print(f"len not equal {len(scr_list)=} != {len(tar_list)=} using smaller size")
#     l1 = len(scr_list) if len(scr_list) < len(tar_list) else len(tar_list)
#     for i in range(l1):
#         print(f"end {i / len(scr_list) * 100:.2f}%")
#         scr = ""
#         tar = ""
#         scr_split = scr_list[i].split(' ')
#         tar_split = tar_list[i].split(' ')
#         while '' in scr_split:
#             scr_split.remove('')
#
#         for scr_ in scr_split:
#             scr += scr_ + " "
#         scr = scr[:-1]
#         while '' in tar_split:
#             tar_split.remove('')
#         for tar_ in tar_split:
#             tar += tar_ + " "
#         tar = tar[:-1]
#         if scr and tar:
#             out.append([scr, tar])
#
#     print(f"len {len(out)=} ")
#
# youtube_data_set_captions.end = out
#   lode_data(model_data, model_data.model_name)
#   write_args(model_data)
#   youtube_data_set_captions.load(model_data.corpus_name)

# youtube_data_set_captions.crate_voc()
#
# youtube_data_set_captions.save("uni_Z1")

voc, pair = youtube_data_set_captions.get()
print(len(pair))

# 16951
# 8000
#
#
#
#
# d = [45, 1.0, 65.0, 0.0001, 6.0, 12000, 10, 250, model_data.model_name, "dot", 512, 6, 6, 0.19, 128, 0, "uni_Z1",
#      voc.export_to_dic()]
model_data.dic_["loadFilename"] = get_new_filename(model_data, 0)
# write_args(model_data)
# load_model_by_name(model_data)

# crate_dic_(model_data, d)

lode_data(model_data, name)

# youtube_data_set_captions.load(model_data.corpus_name)
# v, p = youtube_data_set_captions.get()
# print(v.num_words, p[:5])
# model_data.loadFilename = get_new_filename(model_data, 0)
#
write_args(model_data)

model_data.visual()

# load_model_by_name(model_data)

# talk(model_data)
read_args(model_data)

save_data(model_data, name)

# talk(model_data)

load_model(model_data)

train_model(model_data, pair)

# name = "cb_model"
# model_data_0 = ModelsData("model0_cb_model")
#
##crate_dic(model_data_0)
##write_args(model_data_0)
##corpus_loader(model_data_0)
##load_model(model_data_0)
##
# lode_data(model_data_0, name=name, encoding="utf-8")
# write_args(model_data_0)
# corpus_loader(model_data_0)
# load_model_by_name(model_data_0)  # if model_data_0.dic_["voc"] else load_model_by_name(model_data_0, True)
#
# model_data_0.visual()
#
##talk(model_data_0)
#
talk_ = get_talk(model_data)
#
from time import sleep

while True:
    x = input(": ")
    i = 0
    imp = x
    alt = ""
    while True:
        i += 1
        if i == 10:
            break
        imp = talk_(imp)
        print(f"Bot _ 1 _ : {imp}  {i}")
        imp = talk_(imp)
        print(f"Bot _ 2 _ : {imp}")
        sleep(2)

# import speech_recognition as sr
# import time
# from gtts import gTTS
# import os
# from playsound import playsound
#
# language = 'en'
# r = sr.Recognizer()
#
# with sr.Microphone() as source:
#    while True:
#        x = input("§")
#        try:
#            print("...")
#            audio = r.listen(source, phrase_time_limit=4)
#            text = r.recognize_google(audio, language="de_DE")
#        except:
#            text = "-----"
#        #text.replace("Martin", "markin")
#        print("-->", text)
#        t = time.time()
#        aout = talk_(text)
#        # tar += text.lower() + " " + aout + " "
#        print(time.time() - t)
#        try:
#
#            print(f"Bot: {aout}")
#            if not input("-"):
#                output = gTTS(text=aout, lang=language)
#                name = f"temp{time.time()}_issa.mp3"
#                try:
#                    output.save(name)
#                    playsound(name)
#                    os.remove(name)
#                except Exception:
#                    print("Error")
#        except Exception:
#            print("AssertionError No text to speak")
