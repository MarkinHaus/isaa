end = []
dc = ["<s>", "<e>", ".", ",", "!", "?", "|", "'", '"', "<", ">", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0" ]
num = 0
for n in range(4):

    all_data = [] # http://www.netzmafia.de/software/wordlists/deutsch.txt
    #print(n)
    with open(f"DataTest\\de_text\\liste\\wL ({n}).txt", "r", encoding='utf-8') as data:
        for line in data.readlines():
            num += 1
            if not "." in line or "#" in line:
                if line not in end:
                    end.append(line.strip().lower())
                    #if num == 3000:
                    #    num = 0
                    #    print(f"{(len(end)  / 470052) * 100:.2f}% | len ::  {len(end)}, 470052 |")
        data.close()
dc += end
with open("DIC3.txt", "a", encoding='utf-8') as data:
    data.write(str(dc))
    data.close()
#for word in end:
#    num += 1
#    if not word in dc:
#        dc.append(word)
#    else:
#        x += 1
#
#
#    if num == 3000:
#        num = 0
#        print(f"{(len(dc) / (len(end) - x)) * 100:.2f}% | len :: {len(dc)} , {len(end)} | {x=}")
#
#with open("DIC.txt", "a", encoding='utf-8') as data:
#    data.write(str(dc))
#    data.close()

    #while True:
    #    if input(": ") in all_data:
    #        print("ja")
    #    else:
    #        i = input("~: ")
    #        if i == "+":
    #            data.write(i)
    #        if i == "x":
    #            break
    #        else:
    #            pass

#end = []
#denddata = open("DataTest\\chat.txt", "a", encoding='utf-8')
#with open(f"DataTest\\conv.txt", "r", encoding='utf-8') as data:
#    i = 0
#    for line in data.readlines():
#        i += 1
#        line_eval = eval(line)
#        j = 0
#        for lin_e in line_eval:
#            j += 1
#            try:
#                thread = lin_e["thread"]
#                for text in thread:
#                    out = text["text"]
#                    if not "<" in out:
#                        denddata.write(out+"\n")
#                denddata.write("\n\n")
#                print(f"Done {j} | { + i }")
#            except:
#                print(f"ERROR {j} | { + i}")

#print(len(end))
#
#
#with open("wls.txt", "a", encoding='utf-8') as data:
#    for i in end:
#        data.write(end+"\n")


#a = "http://pcai056.informatik.uni-leipzig.de/downloads/etc/legacy/Papers/top10000de.txt"
#a = a.split("\n")
##print(a)
#alld =[]
#i = 0
import nltk


#def reform_dev(str_data: str):
#    list_data = str_data.replace("\xa0...", "").replace("\n", " ")\
#        .replace(
#        "Wikipedia", "").replace("synonym", "")
#    new_str = " "
#    list_data = nltk.word_tokenize(list_data)
#    for word in list_data:
#        if word in [".", "!", "?"]:
#            with open("w10.txt", "a", encoding='utf-8') as data:
#                data.write(new_str+word+"\n")
#                new_str = " "
#        else:
#            if len(word) >= 2:
#                if "//" in word:
#                    pass
#                else:
#                    new_str += word + " "
#
#a = """"""
#reform_dev(a)
#

