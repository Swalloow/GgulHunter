
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

from sklearn.cross_validation import cross_val_score

feature_names = ["누적매출액"]
label_name = "누적관객수"

train_X = train[feature_names]
train_y = train[label_name]


# ## Score

# In[4]:

from sklearn.svm import LinearSVR


# In[5]:

clf = LinearSVR(C=1.0, max_iter=100)


# In[6]:

score = cross_val_score(clf, train_X, train_y, scoring='median_absolute_error', cv=5).mean()
print("SVR = {0:.6f}".format(-1.0 * score))


# ## Result
# - baseline : 3090.612921
