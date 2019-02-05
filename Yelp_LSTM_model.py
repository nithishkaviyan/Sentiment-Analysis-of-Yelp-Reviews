#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist


# In[32]:


class StateLSTM(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(StateLSTM,self).__init__()
        
        self.lstm=nn.LSTMCell(in_dim,out_dim)
        self.out_dim=out_dim
        
        self.a=None
        self.c=None
        
    def reset_state(self):
        self.a=None
        self.c=None
        
    def forward(self,x):
        batch=x.data.size()[0]
        if (self.a is None):
            state_size=[batch,self.out_dim]
            self.c=Variable(torch.zeros(state_size)).cuda()
            self.a=Variable(torch.zeros(state_size)).cuda()
            
        self.a,self.c=self.lstm(x,(self.a,self.c))
            
        return self.a
    


# In[58]:


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.d=None
        
    def reset_state(self):
        self.d=None
                
    def forward(self,x,dropout=0.5,train=True):
        if (train==False):
            return x
        if(self.d is None):
            self.d=x.data.new(x.size()).bernoulli_(1-dropout)
        mask=Variable(self.d, requires_grad=False)/(1-dropout)
            
        return mask*x


# In[86]:


class Sentiment(nn.Module):
    def __init__(self,vocab_size,hidden_units):
        super(Sentiment,self).__init__()
        self.embedding=nn.Embedding(vocab_size,hidden_units)
        
        self.lstm1=StateLSTM(hidden_units,hidden_units)
        self.bn_lstm1=nn.BatchNorm1d(hidden_units)
        self.lstm1_do=LockedDropout()
        
        self.l1=nn.Linear(hidden_units,1)
        self.loss=nn.BCEWithLogitsLoss()
        
    def reset_state(self):
        self.lstm1.reset_state()
        self.lstm1_do.reset_state()
        
    def forward(self,x,y,train=True):
        embed = self.embedding(x)
        
        steps=embed.shape[1]
        
        self.reset_state()
        
        output=[]
        for i in range(steps):
            a=self.lstm1(embed[:,i,:])
            a=self.bn_lstm1(a)
            a=self.lstm1_do(a,train=train)
            
            output.append(a)
            
        output=torch.stack(output)
        output=output.permute(1,2,0)
        
        max_pool=nn.MaxPool1d(steps)
        a=max_pool(output)
        a=a.view(a.size(0),-1)
        
        a=self.l1(a)
            
        return self.loss(a[:,0],y), a[:,0]
            


# In[105]:


if __name__=="__main__":
    tor=Variable(torch.LongTensor([[1,2,3,4],[5,6,7,8]])).cuda()
    y=Variable(torch.FloatTensor([1,0])).cuda()
    rnn=Sentiment(8,3)
    rnn.cuda()
    rnn(tor,y)

