

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist



class BOW_model(nn.Module):
    def __init__(self,vocab_size,hidden_units):
        super(BOW_model,self).__init__()
        self.embedding=nn.Embedding(vocab_size,hidden_units)
        self.l1=nn.Linear(hidden_units,hidden_units)
        self.l1_bn=nn.BatchNorm1d(hidden_units)
        self.l2_do=nn.Dropout(p=0.5)
        self.l2=nn.Linear(hidden_units,1)
        
        self.loss=nn.BCEWithLogitsLoss()
        
    def forward(self,inp,labels):
        bow_embed=[]
        for i in range(len(inp)):
            lookup_tensor=Variable(torch.LongTensor(inp[i])).cuda()
            embed=self.embedding(lookup_tensor)
            embed=embed.mean(dim=0)
            bow_embed.append(embed)
        bow_embed=torch.stack(bow_embed,dim=0)
            
        x=F.relu(self.l1_bn(self.l1(bow_embed)))
        x=self.l2(self.l2_do(x))
        
        return self.loss(x[:,0],labels), x[:,0]



if __name__=="__main__":
    tor=[[1,2,3,4],[5,6,7,8]]
    y=Variable(torch.FloatTensor([1,0])).cuda()
    bow=BOW_model(8,3)
    bow.cuda()
    bow(tor,y)




