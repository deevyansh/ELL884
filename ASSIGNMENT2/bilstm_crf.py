from torch import nn
from simple_lstm import SimpleLSTM
from implementation import CRF
from config import *

class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,nb_labels,emb_dim=e1_size,hidden_dim=hid_size):
        super().__init__()
        self.lstm=SimpleLSTM(vocab_size,nb_labels,emb_dim=emb_dim,hidden_dim=hidden_dim)
        self.crf=CRF(nb_labels,1,2,0)## we have still not pass the pad id

    def forward(self,x,mask):
        emission=self.lstm(x)
        score,path=self.crf.viterbi_algorithm(emission,mask)
        return score,path
    
    def loss(self,x,y,mask):
        emissions=self.lstm(x)
        nll=self.crf.forward(emissions,y,mask=mask)
        return nll
    
