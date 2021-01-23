#  from BOT.unix.utils import *
from BOT.unix.model import *
from BOT.unix import *
from BOT.unix.get_corpus_y import YouTubeDataSetCaptions
from time import sleep

print(version)

name = "Issa_Z1"

model_data = ModelsData(name)
youtube_data_set_captions = YouTubeDataSetCaptions(name)
youtube_data_set_captions.load("uni_Z1")


lode_data(model_data, name)

write_args(model_data)
model_data.dic_["loadFilename"] = get_new_filename(model_data, 10000)
write_args(model_data)

model_data.visual()

load_model_by_name(model_data)

talk(model_data)

talk_ = get_talk(model_data)


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
#
# print(f"> Hello \n Bot: {talk_('Hello')}")
# sleep(3)
# print(f"> Are u a live \n Bot: {talk_('Are u a live!')}")
# sleep(3)
# print(f"> u are a live\n Bot: {talk_('u are a live!')}")
# sleep(3)
# print(f"> Are u a live\n Bot: {talk_('Are u a live?')}")

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
