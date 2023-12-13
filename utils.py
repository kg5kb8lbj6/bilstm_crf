



# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id["<unk>"] = len(word2id)
    word2id["<pad>"] = len(word2id)
    tag2id["<unk>"] = len(tag2id)
    tag2id["<pad>"] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id["<start>"] = len(word2id)
        word2id["<end>"] = len(word2id)
        tag2id["<start>"] = len(tag2id)
        tag2id["<end>"] = len(tag2id)
    return word2id, tag2id
    
    
    
def prepocess_data_for_lstmcrf(word_lists, tag_lists, test = False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test: # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")
    return word_lists, tag_lists

def sort_by_lengts(word_lists, tag_lists):
    pairs = list(
        zip(word_lists, tag_lists)
    )
    
    indices = sorted(
        range(len(pairs)),
        key = lambda k: len(pairs[k][0]),
        reverse = True
    )
    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))
    return word_lists, tag_lists, indices