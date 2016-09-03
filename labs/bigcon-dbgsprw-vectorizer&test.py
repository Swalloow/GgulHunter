
# coding: utf-8

# # Baseline script of San Francisco Crime Classification
# 
# Baseline script. Hope this helps yours.

# In[129]:

import numpy as np
import pandas as pd


# ## Load Data

# In[74]:

train = pd.read_pickle("../all_df.p")
train.head(3)


# In[75]:

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


# In[76]:

train.shape


# In[77]:

test = pd.read_pickle("../test.p")


# In[78]:

test


# In[79]:

vector_columns = test.columns.drop(['날짜', '영화명', '개봉일'])


# In[80]:

vector_columns


# In[81]:

d_list = []
for each in train.iterrows():
    row = ""
    for column in vector_columns.values :
        if type(each[1][column]) == str :
            row = row + each[1][column] + " "
    d_list.append(row)


# In[82]:

y_list = train["누적관객수"].values


# In[83]:

test_d_list = []
for each in test.iterrows():
    row = ""
    for column in vector_columns.values :
        if type(each[1][column]) == str :
            row = row + each[1][column] + " "
    test_d_list.append(row)


# In[84]:

test_d_list


# ## Vectorize
# 

# In[85]:

from sklearn.feature_extraction.text import TfidfVectorizer


# In[86]:

vectorizer = TfidfVectorizer()
x_list = vectorizer.fit_transform(d_list)


# In[87]:

x_list


# In[88]:

x_list.shape


# In[89]:

x_list._shape = (x_list.shape[0], x_list.shape[1] + 1)


# In[90]:

train["상영기간"] = train["상영기간"].astype('timedelta64[D]').astype(int)


# In[91]:

train.iloc[0]['상영기간']


# In[92]:

x_list.shape[0]


# In[93]:

train.shape


# In[ ]:




# In[94]:

days = train['상영기간'].values


# In[95]:

for i in range(x_list.shape[0]) :
    x_list[i, x_list.shape[1] - 1] = days[i]


# ## CV

# In[ ]:

train['누적관객수']


# In[99]:

def RMSE(y, ypred):
    rmse = np.sqrt(np.mean((y - ypred) ** 2))
    return rmse


# In[118]:


from sklearn.tree import DecisionTreeRegressor
model_rfr = DecisionTreeRegressor()
model_rfr.fit(x_list, y_list)


# In[102]:


from sklearn.ensemble import RandomForestRegressor
model_rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model_rfr.fit(x_list, y_list)


# In[119]:

predict_values = model_rfr.predict(x_list)


# In[120]:

error_rfr = RMSE(y_list, predict_values)


# In[121]:

error_rfr


# In[122]:

test


# In[123]:

vectorizer.vocabulary_


# In[124]:

test_x_list = vectorizer.transform(test_d_list)


# In[125]:

test_x_list._shape = (3, 9148)


# In[126]:

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


# In[127]:

for i in range(test_x_list.shape[0]) :
    test_x_list[i, test_x_list.shape[1] - 1] = test_days[i]


# In[128]:

model_rfr.predict(test_x_list)


# In[ ]:



