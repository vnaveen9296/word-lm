import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Network(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout_prob=0.5):
        super(Network, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        # embeddinng layer
        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # lstm
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # droput
        self.dropout = nn.Dropout(self.dropout_prob)

        # linear
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
    

    def forward(self, x, hidden):
        embed_out = self.embeds(x)

        lstm_out, hidden = self.lstm(embed_out, hidden)

        out = self.dropout(lstm_out)

        out = out.reshape(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    
    def init_hidden(self, batch_size):
        '''initialize hidden state and cell state of lstm'''
        # Shape of h and c: [num_layers x batch_size x hidden_dim]
        weight = next(self.parameters()).data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
    

        