import torch
import torch.nn as nn
from net import Network
from data import Dataset
import pdb
import torch.nn.functional as F
import random
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Predictive typing. Given a sequence of words, predict the possible word
def predictive_typing(model, word_to_ix, ix_to_word, seed="it is"):
    model.eval()

    # initialize the hiddent state and cell state
    batch_size = 1
    h = model.init_hidden(batch_size)

    toks = seed.split()
    for t in toks:
        next_tokens, h = predict(model, t, h, word_to_ix, ix_to_word)

    return next_tokens


# Predict the prob dist of next word and return top 3 words
def predict(model, token, h, word_to_ix, ix_to_word):
    input = torch.tensor(word_to_ix[token]).long()
    input = input.view(1, -1)
    input = input.to(device)

    h = tuple([each.data for each in h])

    # forward pass
    output, h = model(input, h)

    # get token probabilities
    probs = F.softmax(output, dim=1)

    # sort the probabilities and get the indices that correspond to max probs
    indices = torch.argsort(output, dim=1)

    # consider top 3 indices
    indices = indices[:, -3:]
    indices = indices.view(-1)
    # get the words
    tokens = [ix_to_word[index.item()] for index in indices]
    return tokens, h


def load_vocab(infile):
    word_to_ix = {}
    with open(infile, encoding='utf-8') as fin:
        for i, word in enumerate(fin.readlines()):
            word = word.strip()
            word_to_ix[word] = i

    return word_to_ix

if __name__ == '__main__':
    # load the model
    word_to_ix = load_vocab("vocab.txt")
    ix_to_word = {index: word for word, index in word_to_ix.items()}
    vocab_size = len(word_to_ix)
    embedding_dim = 10
    hidden_dim = 20

    model = Network(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load('temp.pth', map_location=torch.device(device)))
    model.to(device)
    print(model)

    text_in = ' '.join(sys.argv[1:])
    if text_in == "":
        text_in = "నేను బయటకి"
    
    text_out = predictive_typing(model, word_to_ix, ix_to_word, text_in)
    print('text_in: ',  text_in)
    print('Top 3 possible words for next token: ')
    for token in text_out[-1::-1]:
        print('  ', token)


