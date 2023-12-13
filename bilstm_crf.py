import torch
from models.config import LSTMConfig,TrainingConfig
from models.biltsm import BiLSTM
import torch.nn as nn

import torch.optim as optim
from utils import sort_by_lengts


        
        
        
class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            """
            
    
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        
        # 模型的参数
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size
        self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None
        
        
    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengts(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengts(
            dev_word_lists, dev_tag_lists)
        B = self.batch_size
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                pass
        
        
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)
        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1 / out_size
        )
    
    def forward(self, sents_tensor, lengths):
        emission = self.bilstm(sents_tensor) 
        # [b,l,out_size]
        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores
    
    

        
