from data import build_corpus

def main():
    print("读取数据.....")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    print(word2id, len(word2id))


if __name__ == "__main__":
    main()
