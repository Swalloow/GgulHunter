
# coding: utf-8

# # Preprocessing csv files

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

filename = "14.08.01-08.07.xlsx.xlsx"
new_filename = "a-14.08.01-08.07.csv"


# In[3]:

train = pd.read_excel("futures_data/" + filename)
train.head()


# ## Preprocessing Excel files

# In[4]:

new_colums = ['순위',
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


# In[5]:

train.columns = new_colums
train.head()


# In[6]:

train = train.dropna(how='all')
train


# In[7]:

train.dtypes


# ## Test - Split and extract dates

# In[8]:

# train['순위'] = train['순위'].astype(str)


# In[9]:

# train['순위'] = train['순위'].apply(lambda x: x.split(' '))
# train.loc[train['순위'].str[0]=='2014년']


# ## Insert Dates column

# In[10]:

train.loc[train['순위']=='합계'].index.values


# In[11]:

train.insert(0, '날짜', '2014년')
train.head(3)


# In[12]:

train.loc[0:69, '날짜'] = '2014-08-07-목'
train.loc[70:151, '날짜'] = '2014-08-06-수'
train.loc[152:234, '날짜'] = '2014-08-05-화'
train.loc[235:304, '날짜'] = '2014-08-04-월'
train.loc[305:373, '날짜'] = '2014-08-03-일'
train.loc[374:445, '날짜'] = '2014-08-02-토'
train.loc[446:522, '날짜'] = '2014-08-01-금'

train.head(3)


# In[13]:

train.loc[train['순위']=='일별 박스오피스 검색 리스트'].index.values[0]


# In[14]:

# train.drop(train['순위']=='순위')
# train.drop(train['순위']=='합계')
train = train.drop(train.loc[train['순위']=='일별 박스오피스 검색 리스트'].index.values[0:])
train = train.drop(train.loc[train['순위']=='순위'].index.values[0:])
train = train.drop(train.loc[train['순위']=='합계'].index.values[0:])
train = train.drop(train.loc[train['매출액 점유율']=='점유율 '].index.values[0:])

train.head(3)


# In[15]:

# train[0:(train.loc[train['순위'].str[0]=='합계'].index.values[0])]
# train.dropna(subset='순위')


# In[16]:

# train['날짜'] = train[0:(train.loc[train['순위'].str[0]=='합계'].index.values[0])]


# In[17]:

# train['순위'] = train_df['순위'].apply(lambda x: clean_file(x))


# ## Output

# In[18]:

train.to_csv(new_filename)


# 
# ---
# # Reset index  - junwoo -

# In[55]:

#train = train.reset_index() # 전체 엑셀파일 데이터 다 모으고 하면됌
#train = train.drop('index',axis=1)


# # Find NaN data and delete row

# In[50]:

train.info() # 보니깐 482개중에 6개가 중간에 끼인 날짜정보칸임 476개만 진짜


# In[51]:

index=pd.isnull(train['영화명'])
for i in range(1,len(index)):
    if index[i]==True :
        print (i)


# In[52]:

train.ix[282]


# In[54]:

train = train.dropna(thresh=10) # NAN값이 10개 이상인 row 삭제
train.info()


# # NaN Data
#  - 개봉일 - 전부 합치고 가장 처음등장한 날짜를 개봉일로 채워주면 될듯 
#  - 대표국적 -
#  - 국적 - 
#  - 제작사 - 
#  - 배급사 - 
#  - 등급 - 
#  - 장르 - 
#  - 감독 -
#  - 배우 - (나머진 고민좀 해봐야할듯)

# In[ ]:



