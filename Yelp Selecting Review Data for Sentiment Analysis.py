
import json
import pandas as pd

#Path for json file
path=open("yelp_academic_dataset_review.json",'r',encoding='utf8')


##Read json file
data_list=[]
for i,x in enumerate(path):
    data_list.append(json.loads(x))


##Convert list to pandas DataFrame
data_reviews=pd.DataFrame.from_dict(data_list,orient='columns')


##Read selected users file
user_data=pd.read_csv('select_users.csv',sep=',')

user_id=user_data['user_id']


select_review=data_reviews[data_reviews['user_id'].isin(user_id)].reset_index(drop=True)

select_review=select_review[['user_id','review_id','business_id','text','stars']]


##Read selected users_neg file
user_data_neg=pd.read_csv('select_users_neg.csv',sep=',')

user_id_neg=user_data_neg['user_id']


select_reviews_12=data_reviews[(data_reviews.stars==1) | (data_reviews.stars==2)].reset_index(drop=True)


user_id_neg=user_data_neg[user_data_neg.review_count > 310]['user_id'].reset_index(drop=True)


select_reviews_12=select_reviews_12[select_reviews_12['user_id'].isin(user_id_neg)]
select_reviews_12=select_reviews_12[['user_id','review_id','business_id','text','stars']]


select_reviews=pd.concat([select_review,select_reviews_12],ignore_index=True)


##Count of number of stars in sample reviews
print(select_reviews.stars.value_counts())


##Drop reviews with stars 3 since its neither a positive nor a negative review
select_reviews=select_reviews[~(select_reviews.stars==3)]



##Write select_reviews to a csv file
select_reviews.to_csv('select_reviews.csv')


##Clean reviews and store in a text file
with open('select_reviews.txt','w', encoding='utf-8') as f:
    for text in select_reviews.text:
        temp_text=text.replace('\n',' ')
        f.write(temp_text)
        f.write('\n')

