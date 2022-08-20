import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='Ciao', help='TRAIN DATASET:\n \
        Ciao,CiaoDVD,douban,Epinions,FilmTrust,Yelp')
args = parser.parse_args()


ratings_dir = './Data/' + args.dataset + '/ratings.txt'
user_list_dir = './Data/' + args.dataset + '/user_list.txt'
item_list_dir = './Data/' + args.dataset + '/item_list.txt'
train_dir = './Data/' + args.dataset + '/train.txt'
test_dir = './Data/' + args.dataset + '/test.txt'

if args.dataset == 'CiaoDVD' or args.dataset == 'Yelp':
    sep = '\t'
else:
    sep = ' '

df1 = pd.read_csv(ratings_dir,sep=sep,names=['userid','itemid','rating'])
user_item_dict = {}
for row in df1.itertuples():
    #if float(row.rating) > 3.0:
    if user_item_dict.get(row.userid) is None:
        user_item_dict[row.userid] = []
    user_item_dict[row.userid].append(row.itemid)

i,j=0,0
user_map = {}
item_map = {}
data = []

for k,v in user_item_dict.items():
    if len(v) > 0:
        if k not in user_map:
            user_map[k] = i
            i += 1
        for m in v:
            if m not in item_map:
                item_map[m] = j
                j += 1
            data.append([user_map[k],item_map[m]])

user_list=[]
item_list=[]
for k,v in user_map.items():
    user_list.append([k,v])
for k,v in item_map.items():
    item_list.append([k,v])

df_user_list = pd.DataFrame(user_list,columns=["org_id","remap_id"])
df_user_list.to_csv(user_list_dir,index=0,sep=' ')
print('user_list is writed to '+user_list_dir)
df_item_list = pd.DataFrame(item_list,columns=["org_id","remap_id"])
df_item_list.to_csv(item_list_dir,index=0,sep=' ')
print('item_list is writed to '+item_list_dir)

df2 = pd.DataFrame(data,columns=['userid','itemid'])
train_data,test_data = train_test_split(df2,test_size=0.2,random_state=42)
train = {}
test = {}

for row in train_data.itertuples():
    if train.get(row.userid) is None:
        train[row.userid] = []
    train[row.userid].append(row.itemid)
with open(train_dir,'w') as f:
    for k,v in train.items():
        f.write(str(k))
        for m in v:
            f.write(' ')
            f.write(str(m))
        f.write('\n')
print("train data is writed to "+train_dir)

for row in test_data.itertuples():
    if test.get(row.userid) is None:
        test[row.userid] = []
    test[row.userid].append(row.itemid)
with open(test_dir,'w') as f:
    for k,v in test.items():
        f.write(str(k))
        for m in v:
            f.write(' ')
            f.write(str(m))
        f.write('\n')
print("test data is writed to "+test_dir)