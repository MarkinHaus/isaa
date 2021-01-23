# imports
from ..unix.utils import * #Voc, EncoderRNN, LuongAttnDecoderRNN, loadPrepareData, loadPrepareData_, trainIters, states, \
    #GreedySearchDecoder, evaluateInput

from BOT.unix.utils import Voc
import torch
import torch.nn as nn
from torch import optim
import os
from io import open


class ModelsData:

    def __init__(self, name):
        self.model_name = name
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token
        self.MIN_COUNT = 3
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cpu")  # "cuda" if self.USE_CUDA else

        self.save_dir = os.path.join("data", "save")

        self.MAX_LENGTH: int = 0  # 10
        self.teacher_forcing_ratio: int = 0   # 1.0
        self.clip: int = 0   # 50.0
        self.learning_rate: float = 0.0   # 0.0001
        self.decoder_learning_ratio: int = 0  # 5.0
        self.n_iteration: int = 0  # 4000
        self.print_every: int = 0  # 1
        self.save_every: int = 0  # 500

        self.attn_model: str = "0"  # 'dot'

        self.hidden_size: int = 0  # 500
        self.encoder_n_layers: int = 0  # 2
        self.decoder_n_layers: int = 0  # 2
        self.dropout: float = 0.0  # 0.1
        self.batch_size: int = 0  # 64
        self.loadFilename: bool = False  # None
        self.checkpoint_iter: int = 0  # 4000

        self.start_iteration: int = 1

        self.corpus_name: str = "0"  # "cornell movie-dialogs corpus"
        self.corpus: str = "0"

        self.embedding = None  #: nn.Embedding
        self.encoder = None  #: EncoderRNN
        self.decoder = None  #: LuongAttnDecoderRNN
        self.voc = Voc("X")  #: Voc

        self.datafile: str = "0"

        self.encoder_optimizer_sd: dict = {}
        self.decoder_optimizer_sd: dict = {}
        self.dic_: dict = {}

    def print(self):
        print(self.model_name)

    def visual(self):
        print_dic(self)


def corpus_loader(all_data):
    all_data.corpus = os.path.join("data\\corpus", all_data.corpus_name)
    all_data.datafile = os.path.join(all_data.corpus, f"{all_data.corpus_name}_lines.txt")


def crate_dic_(all_data, dic__):
    if not len(dic__) == 18:
        raise IndexError(f"{len(dic__)=} != 17 !!")
# [45,1.0,65.0,0.0001,6.0,12000,10,500,all_data.model_name,"dot",512,6,6,0.19,128,0,"y_test",voc]
    all_data.dic_["MAX_LENGTH"] = dic__[0]
    all_data.dic_["teacher_forcing_ratio"] = dic__[1]
    all_data.dic_["clip"] = dic__[2]
    all_data.dic_["learning_rate"] = dic__[3]
    all_data.dic_["decoder_learning_ratio"] = dic__[4]
    all_data.dic_["n_iteration"] = dic__[5]
    all_data.dic_["print_every"] = dic__[6]
    all_data.dic_["save_every"] = dic__[7]
    all_data.dic_["model_name"] = dic__[8]
    all_data.dic_["attn_model"] = dic__[9]
    all_data.dic_["hidden_size"] = dic__[10]
    all_data.dic_["encoder_n_layers"] = dic__[11]
    all_data.dic_["decoder_n_layers"] = dic__[12]
    all_data.dic_["dropout"] = dic__[13]
    all_data.dic_["batch_size"] = dic__[14]
    all_data.dic_["checkpoint_iter"] = dic__[15]
    all_data.dic_["corpus_name"] = dic__[16]
    all_data.dic_["voc"] = dic__[17]


def write_args(all_data):
    all_data.MAX_LENGTH = all_data.dic_["MAX_LENGTH"]
    all_data.teacher_forcing_ratio = all_data.dic_["teacher_forcing_ratio"]
    all_data.clip = all_data.dic_["clip"]
    all_data.learning_rate = all_data.dic_["learning_rate"]
    all_data.decoder_learning_ratio = all_data.dic_["decoder_learning_ratio"]
    all_data.n_iteration = all_data.dic_["n_iteration"]
    all_data.print_every = all_data.dic_["print_every"]
    all_data.save_every = all_data.dic_["save_every"]
    all_data.attn_model = all_data.dic_["attn_model"]
    all_data.hidden_size = all_data.dic_["hidden_size"]
    all_data.encoder_n_layers = all_data.dic_["encoder_n_layers"]
    all_data.decoder_n_layers = all_data.dic_["decoder_n_layers"]
    all_data.dropout = all_data.dic_["dropout"]
    all_data.batch_size = all_data.dic_["batch_size"]
    all_data.loadFilename = all_data.dic_["loadFilename"]
    all_data.checkpoint_iter = all_data.dic_["checkpoint_iter"]
    all_data.corpus_name = all_data.dic_["corpus_name"]
    Voc.import_fom_dic(all_data.voc, all_data.dic_["voc"])


def read_args(all_data):
    all_data.dic_["MAX_LENGTH"] = all_data.MAX_LENGTH
    all_data.dic_["teacher_forcing_ratio"] = all_data.teacher_forcing_ratio
    all_data.dic_["clip"] = all_data.clip
    all_data.dic_["learning_rate"] = all_data.learning_rate
    all_data.dic_["decoder_learning_ratio"] = all_data.decoder_learning_ratio
    all_data.dic_["n_iteration"] = all_data.n_iteration
    all_data.dic_["print_every"] = all_data.print_every
    all_data.dic_["save_every"] = all_data.save_every
    all_data.dic_["model_name"] = all_data.model_name
    all_data.dic_["attn_model"] = all_data.attn_model
    all_data.dic_["hidden_size"] = all_data.hidden_size
    all_data.dic_["encoder_n_layers"] = all_data.encoder_n_layers
    all_data.dic_["decoder_n_layers"] = all_data.decoder_n_layers
    all_data.dic_["dropout"] = all_data.dropout
    all_data.dic_["batch_size"] = all_data.batch_size
    all_data.dic_["loadFilename"] = all_data.loadFilename
    all_data.dic_["checkpoint_iter"] = all_data.checkpoint_iter
    all_data.dic_["corpus_name"] = all_data.corpus_name
    all_data.dic_["voc"] = all_data.voc.export_to_dic()


def lode_data(all_data, name, encoding="utf-8"):
    all_data.dic_ = eval(open(f"data\\forms\\{name}.dic", "r", encoding=encoding).read())


def save_data(all_data, name, encoding="utf-8"):
    open(f"data\\forms\\{name}.dic", "w", encoding=encoding).write(str(all_data.dic_))


def load_model(all_data, VOC: bool = False):
    if VOC:
        all_data.voc, pairs = loadPrepareData(all_data.corpus_name, all_data.datafile)
        all_data.dic_["voc"] = all_data.voc.export_to_dic()

    print('\nBuilding encoder and decoder ...')

    all_data.embedding = nn.Embedding(all_data.voc.num_words, all_data.hidden_size)

    # Initialize encoder & decoder models
    all_data.encoder = EncoderRNN(all_data.hidden_size, all_data.embedding, all_data.encoder_n_layers, all_data.dropout)
    all_data.decoder = LuongAttnDecoderRNN(all_data.attn_model, all_data.embedding, all_data.hidden_size,
                                           all_data.voc.num_words, all_data.decoder_n_layers, all_data.dropout)

    all_data.encoder = all_data.encoder.to(all_data.device)
    all_data.decoder = all_data.decoder.to(all_data.device)
    print('Models built and ready to go!')


def load_model_by_name(all_data, VOC: bool = False):
    if VOC:
        all_data.voc, pairs = loadPrepareData(all_data.corpus_name, all_data.datafile)
        all_data.dic_["voc"] = all_data.voc.export_to_dic()

    checkpoint = torch.load(all_data.loadFilename)

    all_data.start_iteration = checkpoint['iteration'] + 1

    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    all_data.encoder_optimizer_sd = checkpoint['en_opt']
    all_data.decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    all_data.voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    # modelIssa = models.Word2Vec.load('..\\Issa\\isaaB322')
    all_data.embedding = nn.Embedding(all_data.voc.num_words, all_data.hidden_size)  # .from_pretrained(
    # torch.FloatTensor(modelIssa.wv.vectors))
    all_data.embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    all_data.encoder = EncoderRNN(all_data.hidden_size, all_data.embedding, all_data.encoder_n_layers, all_data.dropout)
    all_data.decoder = LuongAttnDecoderRNN(all_data.attn_model, all_data.embedding, all_data.hidden_size,
                                           all_data.voc.num_words,
                                           all_data.decoder_n_layers, all_data.dropout)
    all_data.encoder.load_state_dict(encoder_sd)
    all_data.decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    all_data.encoder = all_data.encoder.to(all_data.device)
    all_data.decoder = all_data.decoder.to(all_data.device)
    print('Models built and ready to go!')


def train_by_name(all_data, pairs):
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(all_data.encoder.parameters(), lr=all_data.learning_rate)
    decoder_optimizer = optim.Adam(all_data.decoder.parameters(), lr=all_data.learning_rate * all_data.
                                   decoder_learning_ratio)

    encoder_optimizer.load_state_dict(all_data.encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(all_data.decoder_optimizer_sd)

    run_train(all_data, encoder_optimizer, decoder_optimizer, pairs)


def train_model(all_data, pairs=None):
    all_data.encoder.train()
    all_data.decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(all_data.encoder.parameters(), lr=all_data.learning_rate)
    decoder_optimizer = optim.Adam(all_data.decoder.parameters(), lr=all_data.learning_rate * all_data.
                                   decoder_learning_ratio)

    run_train(all_data, encoder_optimizer, decoder_optimizer, pairs)


def run_train(all_data, encoder_optimizer, decoder_optimizer, pairs):

    states(encoder_optimizer, decoder_optimizer)

    if not pairs:
        pairs = loadPrepareData_(all_data.corpus_name, all_data.datafile)

    # Run training iterations
    print("Starting Training!")
    trainIters(all_data, pairs, encoder_optimizer, decoder_optimizer)


def talk(all_data):
    all_data.encoder.eval()
    all_data.decoder.eval()

    searcher = GreedySearchDecoder(all_data.encoder, all_data.decoder)

    evaluateInput(searcher, all_data.voc)


def get_talk(all_data):
    from ..unix.utils import normalizeString, evaluate
    all_data.encoder.eval()
    all_data.decoder.eval()

    searcher = GreedySearchDecoder(all_data.encoder, all_data.decoder)

    def auto_talk(input_sentence):
        # input_sentence = normalizeString(input_sentence)
        output_words = evaluate(searcher, all_data.voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ' '.join(output_words)

    return auto_talk


def check_dic_(all_data, name):
    lode_data(all_data, name)


def print_dic(all_data):
    s = f"""
\u001b[1m\u001b[4m\u001b[7m----------------MODEL--------------

model_name:\u001b[0m\u001b[1m\u001b[7m     {all_data.dic_["model_name"]}\u001b[4m
\u001b[4m
-----------------------------------
\u001b[0m\u001b[0m\u001b[31m
-> stets:
\u001b[32m
    {all_data.dic_["MAX_LENGTH"]=}
    {all_data.dic_["teacher_forcing_ratio"]=}
    {all_data.dic_["clip"]=}
    {all_data.dic_["learning_rate"]=}
    {all_data.dic_["decoder_learning_ratio"]=}
    {all_data.dic_["n_iteration"]=}
    {all_data.dic_["print_every"]=}
    {all_data.dic_["save_every"]=}
    {all_data.dic_["attn_model"]=}
    {all_data.dic_["hidden_size"]=}
    {all_data.dic_["encoder_n_layers"]=}
    {all_data.dic_["decoder_n_layers"]=}
    {all_data.dic_["dropout"]=}
    {all_data.dic_["batch_size"]=}
    {all_data.dic_["checkpoint_iter"]=}
\u001b[0m\u001b[4m
-----------------------------------
\u001b[0m\u001b[31m
-> voc: \u001b[0m   {all_data.dic_["corpus_name"]=}
\u001b[31m
-> len: \u001b[0m     {all_data.dic_["voc"]['num_words']}
\u001b[4m
-----------------------------------
\u001b[0m\u001b[31m
-> save at :\u001b[0m   \u001b[4m{all_data.dic_["loadFilename"]}
\u001b[0m
    """

    print(s)


def crate_dic(all_data):
    print("\u001b[1m\u001b[4m----------------MODEL--------------")
    print("model_name ->", all_data.model_name)
    #all_data.dic_["model_name"] = input("model_name -> ")
    all_data.dic_["corpus_name"] = input("corpus_name -> ")
    print("\u001b[4m\n-----------------------------------\u001b[0m\u001b[0m\u001b[31m\n-> stets:\u001b[32m\n")
    all_data.dic_["MAX_LENGTH"] = int(input("MAX_LENGTH -> "))
    all_data.dic_["teacher_forcing_ratio"] = float(input("teacher_forcing_ratio [1.0]-> "))
    all_data.dic_["clip"] = float(input("clip [50.0]-> "))
    all_data.dic_["learning_rate"] = float(input("learning_rate [0.0001]-> "))
    all_data.dic_["decoder_learning_ratio"] = float(input("decoder_learning_ratio [5.0]-> "))
    all_data.dic_["n_iteration"] = int(input("n_iteration [4000]-> "))
    all_data.dic_["print_every"] = int(input("print_every [1]-> "))
    all_data.dic_["save_every"] = int(input("save_every [500]-> "))
    all_data.dic_["attn_model"] = input("attn_model [dot, general, concat]-> ")
    all_data.dic_["hidden_size"] = int(input("hidden_size [500]-> "))
    all_data.dic_["encoder_n_layers"] = int(input("encoder_n_layers [2]-> "))
    all_data.dic_["decoder_n_layers"] = int(input("decoder_n_layers [2]-> "))
    all_data.dic_["dropout"] = float(input("dropout [0.1]-> "))
    all_data.dic_["batch_size"] = int(input("batch_size [64]-> "))
    all_data.dic_["checkpoint_iter"] = int(input("checkpoint_iter [0]-> "))

    all_data.dic_["loadFilename"] = os.path.join(all_data.save_dir, all_data.model_name, all_data.corpus_name,
                                                 '{}-{}_{}'.format(all_data.encoder_n_layers, all_data.decoder_n_layers,
                                                                   all_data.hidden_size),
                                                 '{}_checkpoint.tar'.format(all_data.checkpoint_iter))

    print("\u001b[0m")

    # corpus_loader(all_data)
    # all_data.voc, _ = loadPrepareData(all_data.corpus_name, all_data.datafile)
    # all_data.dic_["voc"] = all_data.voc.export_to_dic()


def get_new_filename(all_data, x=None):
    if not x:
        try:
            x = all_data.dic_["checkpoint_iter"]
        except KeyError:
            x = all_data.checkpoint_iter
    else:
        all_data.dic_["checkpoint_iter"] = x
    return os.path.join(all_data.save_dir, all_data.model_name, all_data.corpus_name,
                                                 '{}-{}_{}'.format(all_data.encoder_n_layers, all_data.decoder_n_layers,
                                                                   all_data.hidden_size),
                                                 '{}_checkpoint.tar'.format(x))