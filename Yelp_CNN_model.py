import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist



class CNN_model(nn.Module):
  def __init__(self,vocab_size,hidden_units,in_dim,nc=5,s=1,p=0,f=3):
    super(CNN_model,self).__init__()
    self.embed=nn.Embedding(vocab_size,hidden_units)
    self.conv1=nn.Conv2d(1,nc,(f,hidden_units),stride=s,padding=p)
    self.maxpool=nn.MaxPool1d(in_dim)
    self.do=nn.Dropout(p=0.2)
    self.l1=nn.Linear(nc,1)
    self.loss=nn.BCEWithLogitsLoss()
   
    
  def forward(self,x,y):
    embed=self.embed(x)
    embed=embed.view(-1,1,embed.shape[1],embed.shape[2])
    x=self.conv1(embed)
    x=x.squeeze(-1)
    x=self.maxpool(x)
    x=x.view(-1,x.shape[1]*x.shape[2])
    x=self.do(x)
    x=self.l1(x)
    loss=self.loss(x[:,0],y)
    
    return loss, x[:,0]
    