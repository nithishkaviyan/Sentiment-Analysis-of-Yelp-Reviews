#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from Yelp_LSTM_model import Sentiment


vocab_size=10000

##Load train data
x_train=[]
with io.open('F:/Yelp Data/Processed Data/yelp_train.txt','r',encoding='utf-8') as f:
    for rev in f:
        review=np.asarray(rev.split(),dtype=np.int)
        review[review>vocab_size]=0
        x_train.append(review)

y_train=[]
with io.open('F:/Yelp Data/Processed Data/yelp_train_labels.txt','r',encoding='utf-8') as f:
    for y in f:
        y_train.append(int(y.strip()))

##Load test data
x_test=[]
with io.open('F:/Yelp Data/Processed Data/yelp_test.txt','r',encoding='utf-8') as f:
    for rev in f:
        review=np.asarray(rev.split(),dtype=np.int)
        review[review>vocab_size]=0
        x_test.append(review)
        
y_test=[]
with io.open('F:/Yelp Data/Processed Data/yelp_test_labels.txt','r',encoding='utf-8') as f:
    for y in f:
        y_test.append(int(y.strip()))
        

vocab_size+=1

#model=Sentiment(vocab_size,500)

model = torch.load('lstm.model')
model.cuda()


opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)


num_epochs=5
batch_size=256

test_loss=[]
test_acc=[]

for epoch in range(num_epochs):
    
    model.eval()
    tes_eplos=[]
    tes_acc=0.0
    tes_count=0.0
    time1=time.time()
    
    perm=np.random.permutation(len(y_test))
    
    for i in range(0,len(y_test),batch_size):
        if((i+batch_size)>=len(y_test)):
            X_inp2=[x_test[j] for j in perm[i:]]
            Y_tes=[y_test[j] for j in perm[i:]]
            batch=len(y_test)-i
        
        else:
            X_inp2=[x_test[j] for j in range(i,i+batch_size)]
            Y_tes=[y_test[j] for j in range(i,i+batch_size)]
            batch=batch_size
    
        seq_length=(epoch+1)*50
        
        X_tes=np.zeros((batch,seq_length),dtype=np.int)
        for j in range(batch):
            X_temp=np.asarray(X_inp2[j])
            rl=X_temp.shape[0]
            if(rl<seq_length):
                X_tes[j,0:rl]=X_temp
            else:
                ran=np.random.randint(rl-seq_length+1)
                X_tes[j,:]=X_temp[ran:(ran+seq_length)] 
        data=Variable(torch.LongTensor(X_tes)).cuda()
        target=Variable(torch.FloatTensor(Y_tes)).cuda()
        
        optimizer.zero_grad()
        with torch.no_grad():
            tes_loss,tes_score=model(data,target,train=False)
            
        tes_eplos.append(round(tes_loss.item(),4))
        
        ##Test Accuracy
        pred=(tes_score>=0)
        truth=(target>=0.5)
        acc=pred.eq(truth).sum().cpu().item()
        tes_acc+=acc
        tes_count+=batch
    
    tes_acc/=tes_count
    test_acc.append(tes_acc*100.0)
    test_loss.append(np.mean(tes_eplos))
    
    print("Sequence length: %.2f" % (seq_length)," Test accuracy: %.4f" %(tes_acc*100.0),          " Test loss: %.4f" %(np.mean(tes_eplos)) ," Elapsed time: %.4f" % (time.time()-time1))
    

