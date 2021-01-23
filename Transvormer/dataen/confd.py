from Transvormer.loade_Model import *
import re
import asyncio


def tok_list_in_list(list_, name, dict_):
    s = []
    i = 0
    for scr_ut in list_:
        i += 1
        print(f"{name} {i / len(list_) * 100:.2f}%")
        scr_tok = scr_ut
        for scr_split in scr_ut.split(" "):
            tok, dict_ = etok(scr_split, dict_)
            scr_tok = re.sub(r"\b%s\b" % scr_split, str(tok), scr_tok)
        s.append(scr_tok)
    out_scr = []
    for scr_ut in s:
        out_scr_ = []
        for scr_split in scr_ut.split(" "):
            if scr_split:
                out_scr_.append(int(scr_split))
        out_scr.append(out_scr_)
    return out_scr, dict_


def tok_list_in_list2(list_, name, dict_):
    s = []
    i = 0
    for scr_ut in list_:
        i += 1
        print(f"{name} {i / len(list_) * 100:.2f}%")
        tok, dict_ = etok(scr_ut, dict_)
        s.append(tok)
        if i == 30:
            break
    return s, dict_


def etok(data, dict_):
    if len(data) <= 1:
        return "", dict_
    try:
        tok = int(data)
    except ValueError:
        # print(data)
        tok, dict_ = te2to(data, dict_, add=True)
    return tok[1], dict_


def etok(data, dict_):
    if len(data) <= 1:
        return "", dict_
    try:
        tok = int(data)
    except ValueError:
        # print(data)
        tok, dict_ = te2to(data, dict_, add=True)
    return tok, dict_


def main(dict_):
    train_data = open("dataen\\tarin\\conf.txt", "r", encoding="utf-8").read()
    train_data = train_data.replace("\\n", "")
    train_data = eval(train_data)
    out = []
    scr_list = train_data["s"]
    tar_list = train_data["e"]

    scr_list, dict_ = tok_list_in_list2(scr_list, "scr", dict_)
    tar_list, dict_ = tok_list_in_list2(tar_list, "tar", dict_)

    print(scr_list[:4], " --- s")
    print(tar_list[:4], " --- t")

    if len(scr_list) == len(tar_list):
        l0 = len(scr_list)
        for i in range(l0):
            print(f"end {i / len(scr_list) * 100:.2f}%")
            out.append([scr_list[i], tar_list[i]])
        print(f"len {l0=} ")
    else:
        print(f"len not equal {len(scr_list)=} != {len(tar_list)=} using smaller size")
        l1 = len(scr_list) if len(scr_list) < len(tar_list) else len(tar_list)
        for i in range(l1):
            print(f"end {i / len(scr_list) * 100:.2f}%")
            out.append([scr_list[i], tar_list[i]])

        print(f"len {l1=} ")

    return out, dict_
