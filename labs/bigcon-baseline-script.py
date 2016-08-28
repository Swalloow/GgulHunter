
# coding: utf-8

# # Baseline script of San Francisco Crime Classification
# 
# Baseline script. Hope this helps yours.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[2]:

train = pd.read_pickle("../all_df.p")
train.head(3)


# In[3]:

feature_names = ["누적매출액"]
label_name = "누적관객수"

train_X = train[feature_names]
train_y = train[label_name]

X = np.array(train_X)
Y = np.array(train_y)


# ## Score

# In[4]:

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold


# In[5]:

svr = LinearSVR(C=1.0, max_iter=100)
rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1)


# In[6]:

K = KFold(len(Y), n_folds=5)
scores_svr = cross_val_score(svr, X, Y, scoring='median_absolute_error', cv=K)
scores_rfr = cross_val_score(rfr, X, Y, scoring='median_absolute_error', cv=K)

print("LinearSVR = {0:.6f}".format(-1.0 * scores_svr.mean()))
print("RandomForestRegressor = {0:.6f}".format(-1.0 * scores_rfr.mean()))


# ## Result
# - LinearSVR : 2108.266677
# - RandomForestRegressor : 819.35333
