
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
from torchnlp.nn import LockedDropout


# In[2]:


# load all that we need
dataset = np.load('../dataset/wiki.train.npy')
dataser_val = np.load('../dataset/wiki.valid.npy')
fixtures_pred = np.load('../fixtures/dev_fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/dev_fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/test_fixtures/prediction.npz')  # test
fixtures_gen_test = np.load('../fixtures/test_fixtures/generation.npy')  # test
vocab = np.load('../dataset/vocab.npy')


# In[3]:


class TextDataset(Dataset):

    def __init__(self, text, seq_len = 70):
        text = combine(text)
        n_seq = len(text) // seq_len
        text = text[:n_seq * seq_len]
        self.data = torch.tensor(text).view(-1,seq_len)
    def __getitem__(self,i):
        txt = self.data[i]
        return txt[:-1],txt[1:]
    def __len__(self):
        return self.data.size(0)

def collate(seq_list):
    inputs = torch.cat([s[0].unsqueeze(1) for s in seq_list],dim=1)
    targets = torch.cat([s[1].unsqueeze(1) for s in seq_list],dim=1)
    return inputs,targets

def combine(text):
    result = np.array([])
    for i in dataset:
        result = np.concatenate((result, i))
    return result.astype(int)


# In[4]:


class LanguageModel(nn.Module):
    def __init__(self, charcount):
        super(LanguageModel, self).__init__()
        self.vocab_size = charcount
        self.embed_size = 400 # From Paper
        self.hidden_size = 1150 # From Paper
        self.nlayers = 3 # From Paper
        self.embed_dropout = 0.1
        self.output_dropout = 0.4

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.nlayers, batch_first=True)
        self.scoring = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, seq_batch, hidden=None):
        batch_size = seq_batch.size(0)
        embed = self.embedding(seq_batch)
        output_lstm, hidden = self.rnn(embed, hidden)
        output = output_lstm
        output_lstm_flatten = output.contiguous().view(-1, self.hidden_size)
        output_flatten = self.scoring(output_lstm_flatten)

        return output_flatten.view(batch_size, -1, self.vocab_size), hidden

    def init_weights(self):
        val = 0.1
        self.embedding.weight.data.uniform_(-val, val)
        self.scoring.bias.data.fill_(0)
        self.scoring.weight.data.uniform_(-val, val)


# In[18]:


class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        self.model = model.cuda()
        self.loader = loader
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001, weight_decay=1e-6)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs, self.max_epochs, epoch_loss))

    def train_batch(self, inputs, targets):
        targets = targets.cuda()
        inputs = inputs.cuda()
        self.optimizer.zero_grad()
        output, hidden = self.model(inputs)
        loss = self.criterion(output.view(-1, self.model.vocab_size), targets.view(-1))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def test(self):
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        nll = test_prediction(predictions, fixtures_pred['out'])

        print('[VAL]  Epoch [%d/%d]   NLL: %.4f'
                      % (self.epochs, self.max_epochs, nll))
        return nll


# In[19]:


class TestLanguageModel:
    def prediction(inp, model):
        inp = inp.astype(int)
        inp = torch.tensor(inp).cuda()
        output, _ = model(inp)
        output = output.cpu().detach().numpy()
        return output[:, -1, :]


# In[20]:


NUM_EPOCHS = 40
BATCH_SIZE = 80


# In[22]:


run_id = str(int(time.time()))
model = LanguageModel(len(vocab))
loader = DataLoader(dataset=TextDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


# In[23]:


best_nll = 1e30  # set to super large value at first
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()

