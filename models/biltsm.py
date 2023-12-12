import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, emb_size
        )
        self.bilstm = nn.LSTM(
            emb_size, hidden_size,
            batch_first = True,
            bidirectional = True
        )
        self.lin = nn.Linear(
            2 * hidden_size, 
            out_size
        )
    
    def forward(self, sents_tensor, lengths):
        pass