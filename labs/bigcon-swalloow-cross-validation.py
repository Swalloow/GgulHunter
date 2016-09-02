
# coding: utf-8

# # Cross validation for BigConTest

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_pickle("../all_df.p")
train.head()


# ## Preprocessing

# In[3]:

from datetime import datetime
import re


# In[4]:

train = train.drop(['매출액','매출액 점유율','매출액증감 (전일대비)','매출액증감율 (전일대비)','관객수','관객수증감 (전일대비)',
                    '관객수증감율 (전일대비)','대표국적','국적','제작사','배급사','등급','장르','감독','배우'], axis=1)
train = train.dropna()
train.head()


# In[5]:

train["날짜"] = train["날짜"].apply(lambda date: re.sub('[(년월화수목금토일)]', '', date))
train["날짜"] = train["날짜"].apply(lambda date: date.replace(" ", "-"))
train["날짜"].head(3)


# In[6]:

train["개봉일"] = pd.to_datetime(train["개봉일"], errors='coerce')
train["개봉일"].head()


# In[7]:

train["상영기간"] = pd.to_datetime(train["날짜"]) - pd.to_datetime(train["개봉일"])
train["상영기간"]


# In[8]:

train = train[train["상영기간"] >= '0 days']
train = train[train["상영기간"] < '5000 days']
train["상영기간"] = train["상영기간"].astype('timedelta64[D]').astype(int)
train.head()


# In[17]:

feature_names = ["누적매출액", "상영횟수", "상영기간"]
label_name = "누적관객수"

train_X = train[feature_names]
train_y = train[label_name]


# ## Cross Validation Set

# In[18]:

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold


# In[19]:

def RMSE(y, ypred):
    rmse = np.sqrt(np.mean((y - ypred) ** 2))
    return rmse


# In[20]:

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3)
y_test.head()


# ## Predict

# In[21]:

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor


# In[22]:

model_svr = LinearSVR(C=1.0, max_iter=100)
model_rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model_rfr.fit(X_train, y_train)
model_svr.fit(X_train, y_train)


# In[23]:

y_pred_rfr = model_rfr.predict(X_test)
y_pred_svr = model_svr.predict(X_test)
y_pred_rfr


# In[24]:

error_rfr = RMSE(y_test, y_pred_rfr)
error_svr = RMSE(y_test, y_pred_svr)
print("RandomForest RMSE = {0:.6f}".format(error_rfr))
print("LinearSVR RMSE = {0:.6f}".format(error_svr))


# ## Result
# - RandomForest RMSE = 31817.049237
# - LinearSVR RMSE = 161836.026577
