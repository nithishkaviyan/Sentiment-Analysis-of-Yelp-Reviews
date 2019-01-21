#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pandas as pd


# In[3]:


#Path for json file
path=open("yelp_academic_dataset_review.json",'r',encoding='utf8')


# In[4]:


##Read json file
data_list=[]
for i,x in enumerate(path):
    data_list.append(json.loads(x))


# In[5]:


print(len(data_list))


# In[ ]:





# In[6]:


##Convert list to pandas DataFrame
data_reviews=pd.DataFrame.from_dict(data_list,orient='columns')


# In[7]:


print(data_reviews.stars.value_counts())


# In[8]:


##Read selected users file
user_data=pd.read_csv('select_users.csv',sep=',')


# In[9]:


user_id=user_data['user_id']


# In[10]:


select_review=data_reviews[data_reviews['user_id'].isin(user_id)].reset_index(drop=True)


# In[11]:


select_review=select_review[['user_id','review_id','business_id','text','stars']]


# In[ ]:





# In[12]:


##Read selected users_neg file
user_data_neg=pd.read_csv('select_users_neg.csv',sep=',')


# In[13]:


user_id_neg=user_data_neg['user_id']


# In[14]:


select_reviews_12=data_reviews[(data_reviews.stars==1) | (data_reviews.stars==2)].reset_index(drop=True)


# In[15]:


user_id_neg=user_data_neg[user_data_neg.review_count > 310]['user_id'].reset_index(drop=True)


# In[16]:


select_reviews_12=select_reviews_12[select_reviews_12['user_id'].isin(user_id_neg)]
select_reviews_12=select_reviews_12[['user_id','review_id','business_id','text','stars']]


# In[17]:


select_reviews=pd.concat([select_review,select_reviews_12],ignore_index=True)


# In[ ]:





# In[18]:


##Count of number of stars in sample reviews
print(select_reviews.stars.value_counts())


# In[19]:


##Drop reviews with stars 3 since its neither a positive nor a negative review
select_reviews=select_reviews[~(select_reviews.stars==3)]


# In[ ]:





# In[20]:


##Write select_reviews to a csv file
select_reviews.to_csv('select_reviews.csv')


# In[21]:


##Clean reviews and store in a text file
with open('select_reviews.txt','w', encoding='utf-8') as f:
    for text in select_reviews.text:
        temp_text=text.replace('\n',' ')
        f.write(temp_text)
        f.write('\n')

