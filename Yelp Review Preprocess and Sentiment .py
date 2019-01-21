#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import itertools
import io
import random

import pandas as pd
import numpy as np
import nltk

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


select_reviews=pd.read_csv('F:/Yelp Data/Processed Data/select_reviews.csv',sep=',')


# In[3]:


select_reviews=select_reviews.drop('Unnamed: 0',axis=1)


# In[4]:


select_reviews.stars.value_counts()


# In[5]:


##Convert stars to positive or negative reviews
select_reviews['sentiment']=select_reviews.stars.apply(lambda x: 1 if x>3 else 0)


# In[6]:


select_reviews.sentiment.value_counts()


# In[ ]:





# In[ ]:


##Split data into train and test sets
tes_index=random.sample(range(0,len(select_reviews)-1),20000)
train_index=[i for i in range(len(select_reviews)) if i not in tes_index]


# In[ ]:





# In[ ]:


train_data=select_reviews.loc[train_index,:].reset_index(drop=True)
test_data=select_reviews.loc[tes_index,:].reset_index(drop=True)


# In[ ]:


##Clean train reviews and store in a text file
with open('F:/Yelp Data/Processed Data/x_train.txt','w', encoding='utf-8') as f:
    for text in train_data.text:
        temp_text=text.replace('\n',' ')
        f.write(temp_text)
        f.write('\n')


# In[ ]:


##Clean test reviews and store in a text file
with open('F:/Yelp Data/Processed Data/x_test.txt','w', encoding='utf-8') as f:
    for text in test_data.text:
        temp_text=text.replace('\n',' ')
        f.write(temp_text)
        f.write('\n')


# In[ ]:


y_train=list(train_data['sentiment'])
y_test=list(test_data['sentiment'])


# In[ ]:





# In[13]:


##Tokenize words in each cleaned reviews (train)
x_train=[]
with open('F:/Yelp Data/Processed Data/x_train.txt','r',encoding='utf-8',newline='\n') as f:
    for lines in f:
        lines=lines.lower()
        x_train.append(nltk.word_tokenize(lines))   


# In[14]:


##Tokenize words in each cleaned reviews (test)
x_test=[]
with open('F:/Yelp Data/Processed Data/x_test.txt','r',encoding='utf-8',newline='\n') as f:
    for lines in f:
        lines=lines.lower()
        x_test.append(nltk.word_tokenize(lines))   


# In[ ]:





# In[15]:


##Train data statistics
train_len_review=[len(i) for i in x_train]


# In[16]:


train_len_review=pd.Series(train_len_review)


# In[17]:


train_df_len=pd.concat([train_data.loc[:,['review_id','sentiment']],train_len_review],axis=1)
train_df_len.columns=['Review_id','Sentiment','Review_length']
train_df_len.head()


# In[18]:


##Box Plot of Review Length
sns.set(style='whitegrid')
sns.boxplot(train_df_len['Review_length'])
plt.title('Box plot of Train Review Length')
plt.show()


# In[19]:


##Mean review length
train_df_len['Review_length'].mean()


# In[20]:


##Median review length
train_df_len['Review_length'].median()


# In[21]:


##Review Length descriptive statistics
train_df_len['Review_length'].describe()


# In[ ]:





# In[ ]:





# In[22]:


##Test data statistics
test_len_review=[len(i) for i in x_test]


# In[23]:


test_len_review=pd.Series(test_len_review)


# In[24]:


test_df_len=pd.concat([test_data.loc[:,['review_id','sentiment']],test_len_review],axis=1)
test_df_len.columns=['Review_id','Sentiment','Review_length']
test_df_len.head()


# In[25]:


##Box Plot of Review Length
sns.set(style='whitegrid')
sns.boxplot(test_df_len['Review_length'])
plt.title('Box plot of Test Review Length')
plt.show()


# In[26]:


##Mean review length
test_df_len['Review_length'].mean()


# In[27]:


##Median review length
test_df_len['Review_length'].median()


# In[28]:


##Review Length descriptive statistics
test_df_len['Review_length'].describe()


# In[ ]:





# In[29]:


##Word to id and id to word
all_tokens=itertools.chain.from_iterable(x_train)
word_to_id={token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)


# In[30]:


##Sort the indices by word frequency
x_train_token_ids=[[word_to_id[token] for token in x] for x in x_train]
count=np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token]+=1

indices=np.argsort(-count)

id_to_word=id_to_word[indices]
count=count[indices]


# In[31]:


## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}


# In[32]:


## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]

x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]


# In[33]:


##Update word_to_id with unknown
word_to_id['<unknown>']=-1

for _,i in word_to_id.items():
    word_to_id[_]=i+1


# In[34]:


id_to_word_dict={}

for n,i in enumerate(id_to_word):
    id_to_word_dict[n+1]=i

id_to_word_dict[0]='<unknown>'


# In[35]:


## save dictionary
np.save('F:/Yelp Data/Processed Data/yelp_dictionary.npy',np.asarray(id_to_word))

## save training data to single text file
with io.open('F:/Yelp Data/Processed Data/yelp_train.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

        
## save test data to single text file
with io.open('F:/Yelp Data/Processed Data/yelp_test.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")


# In[54]:


## save training class to a text file
with io.open('F:/Yelp Data/Processed Data/yelp_train_labels.txt','w',encoding='utf-8') as f:
    for i in y_train:
        f.write("%i " % i)
        f.write("\n")
        
##save test class to a text file
with io.open('F:/Yelp Data/Processed Data/yelp_test_labels.txt','w',encoding='utf-8') as f:
    for i in y_test:
        f.write("%i " % i)
        f.write("\n")


# In[ ]:




