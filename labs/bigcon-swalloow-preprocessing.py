
# coding: utf-8

# # Preprocessing csv files

# In[471]:

import numpy as np
import pandas as pd


# ## Load Data

# In[472]:

filename = "14.08.01-08.07.xlsx.xlsx"
new_filename = "a-14.08.01-08.07.csv"


# In[473]:

train = pd.read_excel("futures_data/" + filename)
train.head()


# ## Preprocessing Excel files

# In[474]:

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


# In[475]:

train.columns = new_colums
train.head()


# In[476]:

train = train.dropna(how='all')
train


# In[477]:

train.dtypes


# ## Test - Split and extract dates
# 
# - 정규표현식으로 contain()을 사용하여 날짜 데이터를 뽑아낸다
# - '날짜' 컬럼에 '순위'=='합계' 가 나오는 바로 위 row 까지 날짜 데이터를 넣는다
# - 엑셀은 제일 마지막 날짜부터 일주일 만큼 순서대로 내려옴
# - dropna()를 사용하여 NaN이 포함된 모든 row를 삭제한다

# In[322]:

# train['순위'] = train['순위'].astype(str)


# In[248]:

# train['순위'] = train['순위'].apply(lambda x: x.split(' '))
# train.loc[train['순위'].str[0]=='2014년']


# ## Insert Dates column

# In[478]:

train.loc[train['순위']=='합계'].index.values


# In[479]:

train.insert(0, '날짜', '2014년')
train.head(3)


# In[480]:

train.loc[0:69, '날짜'] = '2014-08-07-목'
train.loc[70:151, '날짜'] = '2014-08-06-수'
train.loc[152:234, '날짜'] = '2014-08-05-화'
train.loc[235:304, '날짜'] = '2014-08-04-월'
train.loc[305:373, '날짜'] = '2014-08-03-일'
train.loc[374:445, '날짜'] = '2014-08-02-토'
train.loc[446:522, '날짜'] = '2014-08-01-금'

train.head(3)


# In[481]:

train.loc[train['순위']=='일별 박스오피스 검색 리스트'].index.values[0]


# In[486]:

# train.drop(train['순위']=='순위')
# train.drop(train['순위']=='합계')
train = train.drop(train.loc[train['순위']=='일별 박스오피스 검색 리스트'].index.values[0:])
train = train.drop(train.loc[train['순위']=='순위'].index.values[0:])
train = train.drop(train.loc[train['순위']=='합계'].index.values[0:])
train = train.drop(train.loc[train['매출액 점유율']=='점유율 '].index.values[0:])

train.head(3)


# In[ ]:

# train[0:(train.loc[train['순위'].str[0]=='합계'].index.values[0])]
# train.dropna(subset='순위')


# In[ ]:

# train['날짜'] = train[0:(train.loc[train['순위'].str[0]=='합계'].index.values[0])]


# In[72]:

# train['순위'] = train_df['순위'].apply(lambda x: clean_file(x))


# ## Output

# In[487]:

train.to_csv(new_filename)

