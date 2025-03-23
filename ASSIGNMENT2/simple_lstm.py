import torch
from torch import nn
from config import *

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, nb_labels, emb_dim=e1_size, hidden_dim=hid_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, nb_labels)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2),
            torch.randn(2, batch_size, self.hidden_dim // 2),
        )

    def forward(self, batch_of_sentences):
        self.hidden = self.init_hidden(batch_of_sentences.shape[0])
        x, self.hidden = self.lstm(batch_of_sentences, self.hidden)
        x = self.hidden2tag(x)
        return x