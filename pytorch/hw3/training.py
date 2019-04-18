
# coding: utf-8

# In[11]:


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


# In[12]:


# load all that we need
dataset = np.load('../dataset/wiki.train.npy')
dataser_val = np.load('../dataset/wiki.valid.npy')
fixtures_pred = np.load('../fixtures/dev_fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/dev_fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/test_fixtures/prediction.npz')  # test
fixtures_gen_test = np.load('../fixtures/test_fixtures/generation.npy')  # test
vocab = np.load('../dataset/vocab.npy')


# In[46]:


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
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.nlayers)
        self.scoring = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Regularizations
        # Embedding Dropout
        self.embedding_dropout = LockedDropout(self.embed_dropout)
        self.outputting_dropout = LockedDropout(self.output_dropout)

        # Weight tying
        self.scoring.weight = self.embedding.weight

        #weight inilization 
        self.init_weights()

    def forward(self, seq_batch):
        batch_size = seq_batch.size(1)
        embed = self.embedding_dropout(self.embedding(seq_batch))
        hidden = None

        output_lstm, hidden = self.rnn(embed, hidden)
        output = self.outputting_dropout(output_lstm)
        output_lstm_flatten = output.view(-1,self.hidden_size)
        output_flatten = self.scoring(output_lstm_flatten)
        return output_flatten.view(-1, batch_size, self.vocab_size), hidden

    def init_weights(self):
        val = 0.1
        self.embedding.weight.data.uniform_(-val, val)
        self.scoring.bias.data.fill_(0)
        self.scoring.weight.data.uniform_(-val, val)


# In[96]:


# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001, weight_decay=1e-6)
        self.criterion = nn.NLLLoss()

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
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        self.optimizer.zero_grad()
        output, hidden = self.model(inputs)
        loss = self.criterion(output.view(-1, self.model.vocab_size), targets.view(-1))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        nll = test_prediction(predictions, fixtures_pred['out'])
        
        generated_logits = TestLanguageModel.generation(fixtures_gen, 20, self.model) # predictions for 20 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 20, self.model) # predictions for 20 words

        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   NLL: %.4f'
                      % (self.epochs, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-test-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


# In[97]:


class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here
            
            :param inp:
            :return: a np.ndarray of logits
        """
        raise NotImplemented

        
    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """        
        raise NotImplemented
        


# In[98]:


# TODO: define other hyperparameters here

NUM_EPOCHS = 40
BATCH_SIZE = 80


# In[99]:


run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)


# In[100]:


model = LanguageModel(len(vocab))
# loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
loader = DataLoader(dataset=TextDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


# In[ ]:


best_nll = 1e30  # set to super large value at first
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch " + 
              str(epoch)+" with NLL: " + str(best_nll))
        trainer.save()


# In[ ]:

# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation NLL')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()


# In[ ]:


# see generated output
print (trainer.generated[-1]) # get last generated output

