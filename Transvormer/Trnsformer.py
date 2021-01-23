from torch.utils.tensorboard import SummaryWriter
from Transvormer.loade_Model import *
from Transvormer.dataen.confd import main
import random
import copy
from time import time


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
            name,
            learning_rate,
    ):
        super(Transformer, self).__init__()

        self.dic = {

            "embedding_size": embedding_size,
            "src_vocab_size": src_vocab_size,
            "trg_vocab_size": trg_vocab_size,
            "src_pad_idx": src_pad_idx,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "forward_expansion": forward_expansion,
            "dropout": dropout,
            "max_len": max_len,
            "device": " ",
            "name": name,
            "nameEL": [],
            "learning_rate": learning_rate
        }
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size).from_pretrained(
            torch.FloatTensor(modelIssa.wv.vectors))
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(src_vocab_size, embedding_size).from_pretrained(
            torch.FloatTensor(modelIssa.wv.vectors))
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
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
        src_seq_length, N = src.shape
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
        # src[src < 0] = src[src < 0] * -1
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        # trg[trg < 0] = trg[trg < 0] * -1
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

    def safe(self):
        print(f"---> Saving m dic = {self.dic}")
        open(f"models\\{self.dic['name']}", "w").write(str(self.dic))


def format_data(train_data_):  # shape [["hallo ...","hallo..."], ...]
    # imp: was ist 8 + 1 | out : <s> 8 + 1 <e>
    # 1 split scr : <s> was ist 8 + 1 <e> | <tar:  1 <e> > <s>
    # 2 split scr : <s> was ist 8 + 1 <e> | <s> <tar: <e> <s> > 8
    # 3 split scr : <s> was ist 8 + 1 <e> | <s> 8 | <tar: <s> 8 > +
    # 4 split scr : <s> was ist 8 + 1 <e> | <s> 8 + | <tar: 8 + > 1
    # 5 split scr : <s> was ist 8 + 1 <e> | <s> 8 + 1 | <tar: + 1 > <e>
    out_ = []
    i_ = 0
    for x_, y_ in train_data_:
        i_ += 1
        print(f"{len(train_data_) - i_} steps ||  {len(out_)} | {i_ / len(train_data_) * 100:.2f}%")
        end = copy.deepcopy(x_)
        for tar_len in y_:
            tar = end[-2:] + [tar_len]
            out_.append([x_, tar])
            end += [tar_len]

    return out_


if __name__ == "__main__":  # https://www.iskysoft.com/de/videobearbeitung/film-untertitel-herunterladen.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelIssa = models.Word2Vec.load('..\\Issa\\isaaB322')
    vocab = lode_ex_v()

    train_data, vocab = main(vocab)  # eval(open("dataen/data3", "r", encoding='utf-8').read())
    # train_data = eval(open("dataen\\all", "r", encoding="utf-8").read())
    open("dataen\\allv", "w", encoding="utf-8").write(str(train_data))
    print(device)
    print(train_data[:10])

    random.shuffle(train_data)

    # train_data = format_data(train_data)
    # print(train_data[:10])
    # train_data = data[0:2500]
    # open("dataen/data_", "w", encoding="utf-8").write(str(train_data))
    # import numpy

    # print(numpy.max(train_data) * 1000)
    load_model = False
    load_name = "first"
    save_model = True
    name = "first1"

    num_epochs = 150
    learning_rate = 3e-5

    src_vocab_size = 32 #len(vocab)  # len(modelIssa.wv.vocab)
    trg_vocab_size = 32 #len(vocab)
    src_pad_idx = 3
    trg_pad_idx = 3
    embed_size = 32
    num_layers = 12
    forward_expansion = 16
    heads = 4
    num_encoder_layers = 8
    num_decoder_layers = 8
    dropout = 0.25

    max_length = 150

    step = 0

    model = Transformer(
        embed_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_length,
        device,
        name,
        learning_rate,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    if load_model:
        load_checkpoint(torch.load("models\\" + load_name), model, optimizer)
    print(f"---------------------{torch.cuda.get_device_name(0)}---------------------")

    mean_loss = 0
    t1 = 0
    t3 = 0
    t4 = 0
    print(f"{len(train_data)=}")
    for epoch in range(num_epochs):
        # print(f"[Epoch {epoch} / {num_epochs}]")

        model.eval()

        model.train()
        losses = []
        i = 0
        t = time()
        for x, y in train_data:
            i += 1
            if x and y and len(x) < 150 and len(y) < 150:

                # target = torch.tensor([y[: -1]]).to(device)
                # target_ = torch.tensor([y[1:]]).to(device)
                # inp_data = torch.tensor([x]).to(device)

                target = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 1., 0., 0., 0.,
                                        0., 2., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 3., 0., 0., 0.]]).to(device)
                target_ = torch.tensor([[0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.4, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1]]).to(device)
                inp_data = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1,
                                          0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.6, 0.1],
                                         [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.6, 0.1,
                                          0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]).to(device)

                print(inp_data.shape, target.shape, target_.shape)
                # print(inp_data, target, target_)

                # target = target.view(len(y)-1, 1)
                # target_ = target_.view(len(y), 1)
                # inp_data = inp_data.view(len(x), 1)

                # print(">--------------------------<")

                # print(inp_data, target, target_)
                # output = model(inp_data.view(len(x), 1), target.view(len(y) - 1, 1)).to(device)
                print(inp_data.view(32, 6).shape, target.view(32, 1).shape)
                output = model(inp_data, target[1:]).to(device)

                output = output.reshape(-1, output.shape[2])
                target_ = target_.reshape(-1)

                optimizer.zero_grad()

                # print(output.shape, target.shape, target_.shape)

                loss = criterion(output, target_).to(device)
                losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1).to(device)
                optimizer.step()
                step += 1
                if i % 50 == 0:
                    t2 = time() - t
                    t1 += t2
                    t3 += t2
                    t5 = (t2 * (len(train_data) / 50) - t1)

                    print(
                        f"{mean_loss} - {((i * (epoch + 1)) / (len(train_data) * num_epochs) * 100):.3f}% "
                        f"; {loss.item()} - {(i / len(train_data)) * 100:.2f}% --> \n\tE: {epoch} / {num_epochs} | "
                        f"time = {pr_time(t2)}"
                        f" ; E = {pr_time(t1)}"
                        f" || total = {pr_time(t3)} ; preE = {pr_time(t5)} ; preT = {pr_time(t4)} \n")
                    t = time()

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model.dic["nameEL"].append(f"models\\_{name}_\\{name}_E{epoch}_L{mean_loss}.pth.tar")
            save_checkpoint(checkpoint, f"models\\{name}_E{epoch}_L{mean_loss}.pth.tar")
            model.safe()
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        t4 = t1 * (num_epochs - 1)
        t1 = 0

    # running on entire test data takes a while
