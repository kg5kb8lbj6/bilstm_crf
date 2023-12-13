import time
from bilstm_crf import BILSTM_Model


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data
    start = time.time()
    vocab_size, out_size = len(word2id), len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)
