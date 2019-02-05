
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

from Yelp_BOW_model import BOW_model

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

model=BOW_model(vocab_size,500)
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
    X=np.array(x_train)[perm]
    Y=np.array(y_train)[perm]
    for i in range(0,len(Y),batch_size):
        if ((i+batch_size)>=len(Y)):
            continue
        else:
            X_tr=list(X[i:i+batch_size])
            Y_tr=Y[i:i+batch_size]
            label=Variable(torch.FloatTensor(Y_tr)).cuda()
            
            ##Set grad to zero
            optimizer.zero_grad()
            loss,score=model(X_tr,label)
            
            ##Backpropagation
            loss.backward()
            
            epoch_loss.append(round(loss.item(),4))
            
            ##Update Step
            optimizer.step()
            
            ##Training Accuracy
            pred=(score>=0)
            truth=(label>=0.5)
            acc=pred.eq(truth).sum().cpu().item()
            
            epoch_acc+=acc
            epoch_count+=batch_size
            
    epoch_acc/=epoch_count
    train_acc.append(epoch_acc*100.0)
    train_loss.append(np.mean(epoch_loss))
    
    model.eval()
    tes_eplos=[]
    tes_acc=0.0
    tes_count=0.0
    
    for i in range(0,len(y_test),batch_size):
        if((i+batch_size)>=len(y_test)):
            X_tes=[x_test[j] for j in range(i,len(y_test))]
            Y_tes=np.asarray([y_test[j] for j in range(i,len(y_test))],dtype=np.int)
            cnt=len(y_test)-i
        else:
            X_tes=[x_test[j] for j in range(i,i+batch_size)]
            Y_tes=np.asarray([y_test[j] for j in range(i,i+batch_size)],dtype=np.int)
            cnt=batch_size
          
        tes_label=Variable(torch.FloatTensor(Y_tes)).cuda()
        with torch.no_grad():
            tes_loss,tes_score=model(X_tes,tes_label)
        
        tes_eplos.append(round(tes_loss.item(),4))
        
        ##Test Accuracy
        pred=(tes_score>=0)
        truth=(tes_label>=0.5)
        acc=pred.eq(truth).sum().cpu().item()
        tes_acc+=acc
        tes_count+=cnt
    
    tes_acc/=tes_count
    test_acc.append(tes_acc*100.0)
    test_loss.append(np.mean(tes_eplos))
            
    
    print("Epoch: "+str(epoch)+" Train Loss: %.4f" %(np.mean(epoch_loss))+          " Train Accuracy: %.4f" % (epoch_acc*100.0)+" Test Loss: %.4f" %np.mean(tes_eplos)+" Test Accuracy: "+str(tes_acc*100.0)          +" Epoch Time: "+str(time.time()-time1))
    
    

