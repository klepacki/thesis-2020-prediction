#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import os

# import file and set paths

file_base = file_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dane/ultimate')

df = pd.read_csv(os.path.join(file_base, 'dane_imp_finale1.csv').replace(os.sep, '/'), sep = ',')
df = df.drop(df.columns[0], axis=1).drop(['state_po', 'county', 'candidate', 'cand_mayor', 'cand_married'],1)

# Variables encoding

to_encode = ['party', 'state']

labs = LabelEncoder()

for encoder in to_encode:
    df[encoder] = labs.fit_transform(df[encoder])  

# Encoded datasets
    
features_enc = df.drop(['candidatevotes', 'year', 'FIPS'], 1)
cand_votes_enc = df.loc[:, 'candidatevotes']

# Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(features_enc, cand_votes_enc, random_state = 31, test_size = 0.2)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf_params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4, 5],
            'oob_score': [True, False],
            'min_impurity_decrease': [0, 0.05, 0.1, 0.15, 0.2]
            }

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, 
            param_distributions = rf_params1, 
            n_iter = 200, cv = 4, verbose=2, random_state=31, n_jobs = -1)

rf_random.fit(X_train, y_train)
rf_random.best_params_

rand_search = pd.DataFrame(rf_random.cv_results_)

# Plot scoring by iteration

ax = sns.lineplot(x = np.arange(1, 201, 1), y = rand_search.loc[:, "mean_test_score"])
ax.set_title('Scoring by iteration')


corr = rand_search.drop(rand_search.columns[0], 1).corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

preds = rf_random.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse

#Using Pearson Correlation for variables of dataset

plt.figure(figsize=(12,10))
cor = df.drop(['cand_married', 'cand_mayor'], 1).corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

