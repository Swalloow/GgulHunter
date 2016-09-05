
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# ## Data road

# In[2]:

train = pd.read_pickle("../all_df.p")
train = train.reset_index() # 전체 엑셀파일 데이터 다 모으고 하면됌
train = train.drop('index',axis=1)
#train.head(3)


# In[51]:

test = pd.read_pickle("../test.p")
#test


# ### 0. 추가되는 feature

# In[4]:

added_feature = []


# ## preprocessing
# ### 1. date 년원일 분할

# In[5]:

from datetime import datetime
def divide_dates(train):
    total_count = train.shape[0]
    count = 0

    dates_data = []

    for index, row in train['날짜'].iteritems():
        count = count + 1
    
        if count % 10000 == 0:
            print("processing... {0}/{1}".format(count,total_count))
        
        
        date = datetime.strptime(row[0:len(row)-3],"%Y년 %m월 %d일")
    
        dates_data.append({
                "index":index,
                "Dates-Year": date.year,
                "Dates-Month": date.month,
                "Dates-Day": date.day,
            })
    
    dates_dataframe = pd.DataFrame.from_dict(dates_data).astype('int32')
    dates_dataframe = dates_dataframe.set_index("index")

    dates_columns =["Dates-Year","Dates-Month","Dates-Day"]
    dates_dataframe = dates_dataframe[dates_columns]

    train = pd.concat([train,dates_dataframe],axis=1)
    return train


# In[6]:

train=divide_dates(train)
#train.head(3)


# In[52]:

test = divide_dates(test)
#test


# In[8]:

added_feature.append("Dates-Year")
added_feature.append("Dates-Month")
added_feature.append("Dates-Day")


# ### 2. 상영기간 30이하 삭제 및 상영기간 feature 추가

# In[9]:

import re

train["날짜"] = train["날짜"].apply(lambda date: re.sub('[(년월화수목금토일)]', '', date))
train["날짜"] = train["날짜"].apply(lambda date: date.replace(" ", "-"))
train["날짜"].head(3)

train["개봉일"] = pd.to_datetime(train["개봉일"], errors='coerce')
train["개봉일"].head()

train["상영기간"] = pd.to_datetime(train["날짜"]) - pd.to_datetime(train["개봉일"])
train = train.dropna(subset=['상영기간'])

train = train[train["상영기간"] >= '0 days']
train = train[train["상영기간"] < '30 days']


# In[10]:

train.shape


# In[11]:

added_feature = added_feature + ['상영기간']


# ### 3. 대표국적 30개이하 기타로 통합

# In[12]:

def cut_small_country(data):
    data.loc[(data["대표국적"] != "한국")&
             (data["대표국적"] != "미국")&
             (data["대표국적"] != "일본")&
             (data["대표국적"] != "프랑스")&
             (data["대표국적"] != "기타")&
             (data["대표국적"] != "독일")&
             (data["대표국적"] != "영국")&
             (data["대표국적"] != "이탈리아")&
             (data["대표국적"] != "중국")&
             (data["대표국적"] != "홍콩")&
             (data["대표국적"] != "캐나다")&
             (data["대표국적"] != "포르투갈")&
             (data["대표국적"] != "스웨덴")&
             (data["대표국적"] != "러시아")&
             (data["대표국적"] != "대만")&
             (data["대표국적"] != "스페인"),
             "대표국적"] = "기타"


# In[13]:

cut_small_country(train)


# In[14]:

contry = pd.get_dummies(train["대표국적"])


# In[15]:

def count_one_hot(one_hot_data):
    data = pd.DataFrame(columns=["count","country"])
    for row in one_hot_data.columns:
        data_except_rows = one_hot_data[one_hot_data[row] == 1]
        tmp = pd.DataFrame({"country": [row],
                            "count": data_except_rows[row].count()})
        data=data.append(tmp)
    result = data.sort(["count"],ascending=0)
    result = result.reset_index() # 전체 엑셀파일 데이터 다 모으고 하면됌
    result = result.drop('index',axis=1)
    return result


# In[16]:

result = count_one_hot(contry)
#result


# In[17]:

train = pd.concat([train,pd.get_dummies(train["대표국적"])],axis=1)
#train.head(3)


# In[18]:

train.columns


# In[19]:

added_feature=added_feature + ['기타', '대만',
       '독일', '러시아', '미국', '스웨덴', '스페인', '영국', '이탈리아', '일본', '중국', '캐나다', '프랑스',
       '한국', '홍콩']


# In[20]:

added_feature


# ## 4. vectorize  국적, 제작사, 배급사, 장르, 감독, 배우

# In[21]:

vector_columns = test.columns.drop(['날짜', '영화명', '개봉일','대표국적','등급','Dates-Year', 'Dates-Month', 'Dates-Day'])
#vector_columns = test.columns.drop(['날짜', '영화명', '개봉일'])
vector_columns


# In[22]:

d_list = []
for each in train.iterrows():
    row = ""
    for column in vector_columns.values :
        if type(each[1][column]) == str :
            row = row + each[1][column] + " "
    d_list.append(row)


# In[23]:

y_list = train["누적관객수"].values


# In[24]:

test_d_list = []
for each in test.iterrows():
    row = ""
    for column in vector_columns.values :
        if type(each[1][column]) == str :
            row = row + each[1][column] + " "
    test_d_list.append(row)


# In[25]:

test_d_list


# In[26]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x_list = vectorizer.fit_transform(d_list)
x_list


# In[27]:

# 피처갯수만큼 추가 상영기간,년, 원, 일, 한국, 미국, 일본, 프랑스, 기타, 영국, 독일, 
#                   중국,홍콩,캐나다, 스페인, 스웨덴, 포루투갈,러시아, 대만
x_list._shape = (x_list.shape[0], x_list.shape[1] + len(added_feature)) ## 


# In[28]:

train["상영기간"] = train["상영기간"].astype('timedelta64[D]').astype(int)
train.iloc[0]['상영기간']


# In[29]:

train.shape


# In[30]:

for feature in added_feature:
    index = 1;
    temp = train[feature].values
    for i in range(x_list.shape[0]) :
        x_list[i, x_list.shape[1] - index] = temp[i]
    index = index + 1


# In[31]:

x_list


# ## cv

# In[32]:

def RMSE(y, ypred):
    rmse = np.sqrt(np.mean((y - ypred) ** 2))
    return rmse


# In[33]:

from sklearn.tree import DecisionTreeRegressor
model_rfr = DecisionTreeRegressor()
model_rfr.fit(x_list, y_list)


# In[34]:

from sklearn.ensemble import RandomForestRegressor
model_rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model_rfr.fit(x_list, y_list)


# In[35]:

predict_values = model_rfr.predict(x_list)


# In[36]:

error_rfr = RMSE(y_list, predict_values)
error_rfr


# In[53]:

test


# In[54]:

test = pd.concat([test,pd.get_dummies(["대표국적"])],axis=1)


# In[55]:

for f in ['기타', '대만','독일', '러시아',  '스웨덴', '스페인', '영국', '이탈리아', '일본', '중국', '캐나다', '프랑스','홍콩']:
    test[f]=0


# In[56]:

test


# In[57]:

test_x_list = vectorizer.transform(test_d_list)


# In[58]:

test_x_list.shape


# In[59]:

test_x_list._shape = (3, 9162)


# In[60]:

import re

test["날짜"] = test["날짜"].apply(lambda date: re.sub('[(년월화수목금토일)]', '', date))
test["날짜"] = test["날짜"].apply(lambda date: date.replace(" ", "-"))
test["날짜"].head(3)

test["개봉일"] = pd.to_datetime(test["개봉일"], errors='coerce')
test["개봉일"].head()

test["상영기간"] = pd.to_datetime(test["날짜"]) - pd.to_datetime(test["개봉일"])
test = test.dropna(subset=['상영기간'])

test = test[test["상영기간"] >= '0 days']
test = test[test["상영기간"] < '30 days']
test["상영기간"] = test["상영기간"].astype('timedelta64[D]').astype(int)
test_days = test['상영기간'].values


# In[61]:

test_feature = added_feature


# In[62]:

for feature in test_feature:
    index = 1;
    temp = test[feature].values
    for i in range(test_x_list.shape[0]) :
        test_x_list[i,test_x_list.shape[1] - index] = temp[i]
    index = index + 1


# In[63]:

model_rfr.predict(test_x_list)


# In[ ]:



