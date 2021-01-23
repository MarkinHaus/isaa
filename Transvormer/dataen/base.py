import random
from Transvormer.loade_Model import *


def conf(x):
    return str(x).replace("1", "eins ").\
        replace("2", "zwei ").\
        replace("3", "drei ").\
        replace("4", "vier ").\
        replace("5", "fünf ").\
        replace("6", "sechs ").\
        replace("7", "sieben ").\
        replace("8", "acht ").\
        replace("9", "neun ").\
        replace("0", "null ")


def cond(x):
    return str(x).replace("eins ", "1").\
        replace("zwei ", "2").\
        replace("drei ", "3").\
        replace("vier ", "4").\
        replace("fünf ", "5").\
        replace("sechs ", "6").\
        replace("sieben ", "7").\
        replace("acht ", "8").\
        replace("neun ", "9").\
        replace("null ", "0")


def toka(x):
    return te2to(x, dict_)[0]


def main(ll):
    global dict_
    out = []

    dict_ = lode_ex_v()
    for i in range(ll+1):
        print(f"{i} | { (i/ll)*100:.2f}%")
        x = random.randint(0, 90000)
        y = random.randint(0, 90000)
        z = random.choice(["+ ", "- ", "* ", "/ "])

        if z == "+ ":
            e = x+y

        if z == "- ":
            e = x+y

        if z == "* ":
            e = x+y

        if z == "/ ":
            e = x+y

        out.append([toka(conf(f"was ist {x} {z} {y}")), toka(conf(f"das ergibt ist {e}."))])

    return out

# '<s>', '<e>','<x>','<i>','<t>','<d>','<h>','<b>',';', '.', ',', '!', '?', '|', "'", '"', '<', '>'

