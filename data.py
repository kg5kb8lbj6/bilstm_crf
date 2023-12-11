
import os
def build_corpus(split, make_vocab=True, data_dir="./data"):
    """读取数据"""
    assert split in["train","test", "dev"]
    word_lists, tag_lists = [], []
    with open(
        os.path.join(data_dir, "example." + split),
        'r',
        encoding = "utf-8"
    ) as f:
        word_list, 