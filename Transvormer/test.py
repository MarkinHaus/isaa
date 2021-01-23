from Transvormer.loade_Model import mode


def do(model, text, vocab, tar=None):
    from Transvormer.loade_Model import get_out

    def conf(x):
        return str(x).replace("1", "eins "). \
            replace("2", "zwei "). \
            replace("3", "drei "). \
            replace("4", "vier "). \
            replace("5", "fünf "). \
            replace("6", "sechs "). \
            replace("7", "sieben "). \
            replace("8", "acht "). \
            replace("9", "neun "). \
            replace("0", "null ")

    def cond(x):
        return str(x).replace("eins ", "1"). \
            replace("zwei ", "2"). \
            replace("drei ", "3"). \
            replace("vier ", "4"). \
            replace("fünf ", "5"). \
            replace("sechs ", "6"). \
            replace("sieben ", "7"). \
            replace("acht ", "8"). \
            replace("neun ", "9"). \
            replace("null ", "0")

    def plsm(x):
        return str(x).replace("+", "plus"). \
            replace("-", "minus"). \
            replace("*", "mal"). \
            replace("/", "geteilt"). \
            replace("^", "hoch")

    def eplsm(x):
        return str(x).replace("plus", "+"). \
            replace("minus ", "-"). \
            replace("mal", "*"). \
            replace("geteilt", "/"). \
            replace("hoch", "^")

    print(f"~ {text} ")
    if text != "-----" and text != "":
        scr = conf(text)
        print(f"scr-> {scr} ")
        out = get_out(model, vocab, scr)
        return cond(out)


def do1(model, text, vocab, tar=None):
    from Transvormer.loade_Model import get_out

    print(f"~ {text} ")
    if text != "-----" and text != "":
        scr = text
        print(f"scr-> {scr} ")
        out = get_out(model, vocab, scr)
        return out


mode("t2", sound=True, do=do1, ind=1)
