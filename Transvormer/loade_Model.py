import torch.nn as nn
import torch.optim as optim
from Transvormer.utils import *
from gensim import models
import copy


class Transformer(nn.Module):
    def __init__(
            self,
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device,
            modelIssa,

    ):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size).from_pretrained(
            torch.FloatTensor(modelIssa.wv.vectors))
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(src_vocab_size, embedding_size).from_pretrained(
            torch.FloatTensor(modelIssa.wv.vectors))
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = torch.device(device)
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        try:
            src_seq_length, N, _ = src.shape
        except:
            src_seq_length, N = src.shape
        try:
            trg_seq_length, N, _ = trg.shape
        except:
            trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
                .unsqueeze(1)
                .expand(src_seq_length, N)
                .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(1)
                .expand(trg_seq_length, N)
                .to(self.device)
        )
        src[src < 0] = src[src < 0] * -1
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        trg[trg < 0] = trg[trg < 0] * -1
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


def loade_by_name(name, num=-1):
    dict_ = eval(open(f"models\\_{name}_\\{name}", "r").read())
    dict_["device"] = torch.device("cpu")
    modelIssa = models.Word2Vec.load('..\\Issa\\isaaB322')

    model = Transformer(
        dict_["embedding_size"],
        dict_["src_vocab_size"],
        dict_["trg_vocab_size"],
        dict_["src_pad_idx"],
        dict_["num_heads"],
        dict_["num_encoder_layers"],
        dict_["num_decoder_layers"],
        dict_["forward_expansion"],
        dict_["dropout"],
        dict_["max_len"],
        dict_["device"],
        modelIssa
    ).to(dict_["device"])
    print("=> Loading model Data")
    optimizer = optim.Adam(model.parameters(), lr=dict_["learning_rate"])
    if not "\\" in dict_['nameEL'][num]:
        assert IndexError("models\\ DU NUDEL")
    checkpoint = torch.load(dict_['nameEL'][num])
    print(f"=> Loading {dict_['nameEL'][num]}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("=> Loaded")

    return model, modelIssa


def get_pre(model, scr, tar):
    # print(scr, tar)
    target = torch.tensor([tar]).unsqueeze(1)
    inp_data = torch.tensor([scr])
    output = model(inp_data.view(len(scr), 1), target.view(len(tar), 1))
    return output


def get_out(model, vocab, imp, max_length=20):
    imp = te2to(imp, vocab)[0]
    # tar = te2to(tar, vocab)
    print(imp)
    sentence_tensor = copy.deepcopy(imp)
    # best_guess = copy.deepcopy(imp[-2:])
    outputs = [0, ]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs)
        with torch.no_grad():
            output = get_pre(model, sentence_tensor, outputs)
            # print(output)
        best_guess = output.argmax(2)[-1, :].item()
        # print(best_guess)
        outputs.append(best_guess)

        if i == max_length or best_guess == vocab.index("<e>"):
            # print(vocab[42069])
            break
    # print(outputs)
    return to2te(outputs, vocab)


def mode(name, sound, do, ind=-1):
    import speech_recognition as sr
    import time
    from gtts import gTTS
    import os
    from playsound import playsound
    model, modelIssa = loade_by_name(name, ind)
    vocab = lode_ex_v()
    language = 'de'
    r = sr.Recognizer()
    tar = ""
    tok = ['<s>', '<e>', '<x>', '<i>', '<t>', '<d>', '<h>', '<b>']
    with sr.Microphone() as source:
        while True:
            x = input("§")
            if sound:
                try:
                    print("...")
                    audio = r.listen(source, phrase_time_limit=4)
                    text = r.recognize_google(audio, language="de_DE")

                except:
                    text = "-----"
                text.replace("Martin", "markin")

                print("-->")
                t = time.time()

                aout = do(model, text.lower(), vocab, tar=tar).strip()

                # tar += text.lower() + " " + aout + " "

                print(time.time() - t)
                try:
                    for m in tok:
                        aout = aout.replace(m, "")
                    if aout.strip():
                        aout = "das habe ich nicht verstanden"
                    print(f"Bot: {aout}")
                    output = gTTS(text=aout, lang=language)
                    name = f"temp{time.time()}_issa.mp3"
                    try:
                        output.save(name)
                        playsound(name)
                        os.remove(name)
                    except Exception:
                        print("Error")
                except Exception:
                    print("AssertionError No text to speak")
            else:
                text = x
                print("-->")
                t = time.time()

                aout = do(model, text.lower(), vocab, tar=tar)

                print(time.time() - t, "\n", aout)

                # tar += text.lower() + " " + aout + " "

            if text == "Ende":
                break
