version = "0.0.1"
HELP = """
def load_from_name(name):
    _data = ModelsData(--)
    lode_data(_data, name=name, encoding="utf-8")
    write_args(_data)
    corpus_loader(_data)

    load_model_by_name(_data) if _data.dic_["voc"] else load_model_by_name(_data, True)



def crate_new_from_dic(dic):
    _data = ModelsData(--)
    crate_dic_(_data, dic)
    write_args(_data)
    corpus_loader(_data)
    load_model(_data)

    save_data(_data, name=_data.model_name, encoding="utf-8")



def crate_new():
    _data = ModelsData(--)
    crate_dic(_data)
    write_args(_data)
    corpus_loader(_data)
    load_model(_data)

    save_data(_data, name=_data.model_name, encoding="utf-8")



"""
