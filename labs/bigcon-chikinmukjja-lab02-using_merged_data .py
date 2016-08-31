
# coding: utf-8

# # Baseline script of San Francisco Crime Classification
# 
# Baseline script. Hope this helps yours.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_pickle("merged_data.p")
train.head(3)
train = train.reset_index() # 인데스
train = train.drop('index',axis=1)


# In[3]:

train.head(3)


# In[4]:

pd.get_dummies(train["대표국적"]).head(3)


# In[5]:

pd.get_dummies(train["장르"]).head(3)


# In[6]:

pd.get_dummies(train["국적"]).head(3)


# In[7]:

pd.get_dummies(train["배급사"]).head(3)


# In[8]:

pd.get_dummies(train["제작사"]).head(3)


# In[9]:

pd.get_dummies(train["등급"]).head(3)


# In[40]:

from sklearn.cross_validation import cross_val_score

feature_names = ["순위","Dates-Year","Dates-Month","Dates-Day"]
label_name = "누적관객수"

train_X = train[feature_names]
train_y = train[label_name]


# In[41]:

train_X = pd.concat([train_X,pd.get_dummies(train["대표국적"])],axis=1)
train_X = pd.concat([train_X,pd.get_dummies(train["등급"])],axis=1)
train_X = pd.concat([train_X,pd.get_dummies(train["장르"])],axis=1)
train_X = pd.concat([train_X,pd.get_dummies(train["국적"])],axis=1)
train_X = pd.concat([train_X,pd.get_dummies(train["제작사"])],axis=1)
train_X = pd.concat([train_X,pd.get_dummies(train["배급사"])],axis=1)


# In[42]:

train_X.head(3)


# ## Score

# In[51]:

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor


# In[61]:

clf = LinearSVR(C=1.0, max_iter=100)
rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1,max_depth=10)


# In[62]:

score = cross_val_score(clf, train_X, train_y, scoring='mean_squared_error', cv=5).mean()
print("SVR = {0:.6f}".format(-1.0 * score))


# In[63]:

score = cross_val_score(rfr, train_X, train_y, scoring='mean_squared_error', cv=5,verbose=True).mean()
print("RFR = {0:.6f}".format(-1.0 * score))


# ## Result
