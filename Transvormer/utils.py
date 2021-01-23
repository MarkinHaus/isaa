import torch


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def append_v(list_data, vocab, file="dataen/DICs_.txt"):
    vocab = vocab + list_data
    open(file, "w", encoding="utf-8").write(str(vocab))
    print("=> done")
    return vocab


def lode_ex_v(file="..\\Issa\\DICs_.txt"):
    print("=> Loading vocab..")
    x = eval(open(file, "r", encoding="utf-8").read())
    print("=> done ")
    return x


def te2to(text: str, vocab: list, add=False):
    tok_list = [0]
    if add:
        stm = []
    for word in text.lower().split(" "):
        if word:
            try:
                tok_list.append(vocab.index(word))
            except:
                if add:
                    if word in stm:
                        tok_list.append(len(vocab)+stm.index(word))
                    else:
                        stm.append(word)
                        tok_list.append(len(vocab) + stm.index(word))
                else:
                    tok_list.append(vocab.index("<x>"))
                    print(word)
    tok_list += [1]
    if add and len(stm) > 0:
        print(stm, " -- ADD", )
        vocab = append_v(stm, vocab)
    return tok_list, vocab


def to2te(tok_list: list, vocab: list):
    end = ""
    # print(tok_list)
    for tok in tok_list:
        # print(tok)
        try:
            end += vocab[tok] + " "
        except IndexError:
            end += " |missing| "
    return end


def pr_time(x):
    x = int(x)
    h = int(x/3600)
    m = int(x/60-h*60)
    s = int(x-m*60-h*3600)
    return f"{h}:{m}:{s}"