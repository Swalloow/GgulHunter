
# coding: utf-8

# # Preprocessing csv files

# In[5]:

import numpy as np
import pandas as pd
import os


# ## Load Data

# In[6]:

mypath = "./futures_data"


# In[ ]:

def excel_parsing(excel):
    train = pd.read_excel(mypath + "/" + excel)

    train.insert(0, '날짜', '없엉')
    colum_name = train.columns.values[1]
    train.loc[4,'날짜'] = train.columns.values[1]
    for i in range(5, len(train)) :
        if type(train.loc[i, colum_name]) == int:
            if train.loc[i-1, '날짜'] != '없엉' :
                train.loc[i,'날짜'] = train.loc[i-1, '날짜']
            else :
                train.loc[i,'날짜'] = train.loc[i-5, colum_name]

    train = train[train['날짜'] != '없엉']
    ## Preprocessing Excel files
    new_colums = ['날짜',
                  '순위',
                  '영화명',
                  '개봉일',
                  '매출액',
                  '매출액 점유율',
                  '매출액증감 (전일대비)',
                  '매출액증감율 (전일대비)',
                  '누적매출액',
                  '관객수',
                  '관객수증감 (전일대비)',
                  '관객수증감율 (전일대비)',
                  '누적관객수',
                  '스크린수',
                  '상영횟수',
                  '대표국적',
                  '국적',
                  '제작사',
                  '배급사',
                  '등급',
                  '장르',
                  '감독',
                  '배우'
                 ]
    train.columns = new_colums
    return train


# In[28]:

all_df = pd.DataFrame(columns=['날짜',
                  '순위',
                  '영화명',
                  '개봉일',
                  '매출액',
                  '매출액 점유율',
                  '매출액증감 (전일대비)',
                  '매출액증감율 (전일대비)',
                  '누적매출액',
                  '관객수',
                  '관객수증감 (전일대비)',
                  '관객수증감율 (전일대비)',
                  '누적관객수',
                  '스크린수',
                  '상영횟수',
                  '대표국적',
                  '국적',
                  '제작사',
                  '배급사',
                  '등급',
                  '장르',
                  '감독',
                  '배우'
                 ])
file_names = os.listdir(mypath)
for file_name in file_names :
    print (file_name)
    all_df = all_df.append(excel_parsing(file_name))
    


# In[29]:

all_df


# In[30]:

all_df.to_csv('all_data.csv', index=False)


# In[ ]:



