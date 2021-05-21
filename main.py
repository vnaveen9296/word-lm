import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from net import Network
from data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from predictive_typing import predictive_typing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')


def train(loader, model, optimizer, loss_fn, batch_size, num_epochs=10, clip=1):
    model.train()
    # training loop
    for t in range(1, num_epochs+1):
        total_loss = 0
        h = model.init_hidden(batch_size)
        counter = 0
        for x, y in loader:
            counter += 1
            x, y = x.to(device), y.to(device)
            
            h = tuple([each.data for each in h])
            output, h = model(x, h)

            loss = loss_fn(output, y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch: {t}/{num_epochs}, loss: {total_loss}')


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


if __name__ == '__main__':
    dataset = Dataset('telugu.txt')
    vocab_size = len(dataset.word_to_ix)
    embedding_dim = 10
    hidden_dim = 20

    torch.manual_seed(1234)
    model = Network(vocab_size, embedding_dim, hidden_dim)
    model = model.to(device)
    print(model)

    batch_size = 2
    learning_rate = 0.001
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train(loader, model, optimizer, loss_fn, batch_size, num_epochs=100)


    # save the model
    torch.save(model.state_dict(), "temp.pth")
    print(f'saved the model to temp.pth')

    text_in = "నేను బయటకి"
    text_out = predictive_typing(model, dataset.word_to_ix, dataset.ix_to_word, text_in)
    
    print('text_in: ',  text_in)
    print('Top 3 possible words for next token: ')
    for token in text_out[-1::-1]:
        print('  ', token)
