from pytube import YouTube
from BOT.unix.utils import Voc, normalizeString


class YouTubeDataSetCaptions:

    def __init__(self, name):
        self.name = name
        self.end = []
        self.data = []
        self.voc = Voc(name)

    def append_conv(self, urls, lang="de"):
        for url in urls:
            try:
                myVid = YouTube(url)
                myVidC = myVid.captions.get_by_language_code(lang)
                self.data.append(myVidC.generate_srt_captions())
                print(f"\u001b[1m\u001b[4mName: {myVid.title}\u001b[0m\n")
            except Exception:
                print(f"Error {url}")

    def format(self):
        flip = 0
        out = []
        pair = []
        for data in self.data:
            for line in data.split("\n\n"):
                try:
                    id_, stamp, data_ = line.split("\n")[0], line.split("\n")[1], line.split("\n")[2]
                    print(f"{id_=},     {stamp=}")
                    if flip == 0:
                        pair = [data_]
                        flip = 1
                    if flip == 1:
                        if pair[0] != data_:
                            pair += [data_]
                            flip = 0
                            out.append(pair)
                            pair = []
                        else:
                            flip = 1
                    if '' in data_.split(' '):
                        flip = 0
                        pair = []
                except Exception as e:
                    print(e)
                    pass
        self.end = out

    def crate_voc(self):
        for pair in self.end:
            self.voc.addSentence(pair[0])
            self.voc.addSentence(pair[1])

    def get(self):
        return self.voc, self.end

    def save(self, corpus_name):
        # TODO FileNotFoundError: [Errno 2] No such file or directory
        open(f"data\\corpus\\{corpus_name}\\{self.name}_crp.crp", "w").write(
            str({"voc": self.voc.export_to_dic(), "pairs": self.end}))

    def load(self, corpus_name):
        data = eval(open(f"data\\corpus\\{corpus_name}\\{self.name}_crp.crp", "r").read())
        self.end = data["pairs"]
        Voc.import_fom_dic(self.voc, data["voc"])

# Kev1
# y_mojirClzQEs
# 25
# 1.0
# 55.0
# 0.0001
# 5.0
# 12000
# 10
# 500
# dot
# 500
# 4
# 4
# 0.15
# 64
# 0
