
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

from Yelp_LSTM_model import Sentiment

import os
import time
import sys
import io


vocab_size=10000

##Load train data
x_train=[]
with io.open('yelp_train.txt','r',encoding='utf-8') as f:
    for rev in f:
        review=np.asarray(rev.split(),dtype=np.int)
        review[review>vocab_size]=0
        x_train.append(review)

y_train=[]
with io.open('yelp_train_labels.txt','r',encoding='utf-8') as f:
    for y in f:
        y_train.append(int(y.strip()))



##Load test data
x_test=[]
with io.open('yelp_test.txt','r',encoding='utf-8') as f:
    for rev in f:
        review=np.asarray(rev.split(),dtype=np.int)
        review[review>vocab_size]=0
        x_test.append(review)
        
y_test=[]
with io.open('yelp_test_labels.txt','r',encoding='utf-8') as f:
    for y in f:
        y_test.append(int(y.strip()))



vocab_size+=1

model=Sentiment(vocab_size,500)
model.cuda()



##Optimizer
optimizer=optim.Adam(model.parameters(),lr=0.001)

num_epochs=5
batch_size=256
train_loss=[]
train_acc=[]

test_loss=[]
test_acc=[]

for epoch in range(num_epochs):
    
    model.train()
    
    time1=time.time()
    epoch_loss=[]
    epoch_acc=0.0
    epoch_count=0.0
    
    perm=np.random.permutation(len(x_train))
    
    for i in range(0,len(x_train),batch_size):
        if ((i+batch_size)>=len(x_train)):
            continue
        else:
            X_inp=[x_train[j] for j in perm[i:i+batch_size]]
            seq_length=100
            X_tr=np.zeros((batch_size,seq_length),dtype=np.int)
            for j in range(batch_size):
                X_temp=np.asarray(X_inp[j])
                rl=X_temp.shape[0]
                if(rl<seq_length):
                    X_tr[j,0:rl]=X_temp
                else:
                    ran=np.random.randint(rl-seq_length+1)
                    X_tr[j,:]=X_temp[ran:(ran+seq_length)]
            Y_tr=y_train[perm[i:i+batch_size]]
            
            data=Variable(torch.LongTensor(X_tr)).cuda()
            target=Variable(torch.FloatTensor(Y_tr)).cuda()
            
            optimizer.zero_grad()
            
            loss,score=model(data,target,train=True)
            loss.backward()
            
            epoch_loss.append(round(loss.item(),4))
            
            ##Update
            optimizer.step()
            
            ##Training Accuracy
            pred=(score>=0)
            truth=(target>=0.5)
            acc=pred.eq(truth).sum().cpu().item()
            
            epoch_acc+=acc
            epoch_count+=batch_size
            
    epoch_acc/=epoch_count
    train_acc.append(epoch_acc*100.0)
    train_loss.append(np.mean(epoch_loss))
    
    print("Epoch: "+str(epoch)+" Train Loss: %.4f" %(np.mean(epoch_loss))+          " Train Accuracy: %.4f" % (epoch_acc*100.0)+          " Epoch Time: "+str(time.time()-time1))
    
    if(epoch%3==0):
        model.eval()
        tes_eplos=[]
        tes_acc=0.0
        tes_count=0.0
        
        for i in range(0,len(y_test),batch_size):
            if((i+batch_size)>=len(y_test)):
                X_inp2=[x_test[j] for j in range(i,len(y_test))]
                Y_tes=[y_test[j] for j in range(i,len(y_test))]
                batch=len(y_test)-i
        
            else:
                X_inp2=[x_test[j] for j in range(i,i+batch_size)]
                Y_tes=[y_test[j] for j in range(i,i+batch_size)]
                batch=batch_size
            
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
            
    
        print("Test Loss: %.4f" %np.mean(tes_eplos)+" Test Accuracy: "+str(tes_acc*100.0))
    
        

