
# coding: utf-8

# # Baseline script of San Francisco Crime Classification
# 
# For analysis model and datasets.

# In[1]:

import numpy as np
import pandas as pd


# ## Load Data

# In[9]:

train = pd.read_pickle("../all_df.p")
train.head(3)


# In[30]:

feature_names = ["누적매출액"]
label_name = "누적관객수"

train_X = train[feature_names]
train_y = train[label_name]


# ## Score

# In[31]:

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold


# In[41]:

svr = LinearSVR(C=1.0, max_iter=100)
rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1)


# In[43]:

X = np.array(train_X)
Y = np.array(train_y)

K = KFold(len(Y), n_folds=5)
scores_svr = cross_val_score(svr, X, Y, scoring='median_absolute_error', cv=K)
scores_rfr = cross_val_score(rfr, X, Y, scoring='median_absolute_error', cv=K)
print("LinearSVR = {0:.6f}".format(-1.0 * scores_svr.mean()))
print("RandomForestRegressor = {0:.6f}".format(-1.0 * scores_rfr.mean()))


# In[29]:

score_svr = cross_val_score(svr, train_X, train_y, scoring='median_absolute_error', cv=5).mean()
score_rfr = cross_val_score(rfr, train_X, train_y, scoring='median_absolute_error', cv=5).mean()
score_lr = cross_val_score(lr, train_X, train_y, scoring='median_absolute_error', cv=5).mean()

print("LinearSVR = {0:.6f}".format(-1.0 * score_svr))
print("RandomForestRegressor = {0:.6f}".format(-1.0 * score_rfr))
print("LinearRegression = {0:.6f}".format(-1.0 * score_lr))


# ## Graph

# In[17]:

plt.scatter(train_X, train_y, c='k', label='data')
plt.hold('on')
# plt.plot(train_X, y_svr, c='r', label='Linear SVR')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# ## Result
# - baseline : 3090.612921
