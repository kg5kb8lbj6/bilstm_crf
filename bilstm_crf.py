import torch
from models.config import LSTMConfig
from models.biltsm import BiLSTM
import torch.nn as nn


        
        
        
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
        BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
        
        
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        BiLSTM(vocab_size, emb_size, hidden_size, out_size)