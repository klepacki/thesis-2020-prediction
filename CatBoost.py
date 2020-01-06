#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os

file_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dane/ultimate')

df = pd.read_csv(os.path.join(file_base, 'dane_imp_finale1.csv').replace(os.sep, '/'), sep = ',')
df = df.drop(df.columns[0], axis=1)
df = df.drop(['cand_mayor', 'cand_married', 'candidate', 'state_po', 'county'], 1)


from catboost import CatBoostRegressor, Pool

# Defining the categorical variables

cat_variables = ['party', 'state']

# Splitting the sets

cand_votes = df.loc[:, 'candidatevotes']
features = df.drop('candidatevotes', 1)

X_train, X_test, y_train, y_test = train_test_split(features, cand_votes, random_state = 31, test_size = 0.2)

train_pool = Pool(X_train, y_train, cat_features = cat_variables)
test_pool = Pool(X_test, y_test, cat_features = cat_variables)

# Setting the parameter space

params = {'learning_rate': np.arange(0.01, 0.1, 0.01),
         'iterations': [500, 750, 1000, 1250, 1500, 1750, 2000],
         'depth': [4, 6, 8],
         'l2_leaf_reg': [1, 3, 5, 7],
         'bootstrap_type': ['Bernoulli', 'Poisson'],
         'boosting_type': ['Plain'],
         'min_data_in_leaf': [10, 20, 30],
         'leaf_estimation_method': ['Gradient'],
         'early_stopping_rounds': [50, 75]}

ctboost = CatBoostRegressor(task_type = 'GPU')

model = ctboost.randomized_search(params, X = train_pool, search_by_train_test_split = False, cv = 3, plot = True, n_iter = 150)

# Get the best params

model

ctboost1 = CatBoostRegressor(task_type = 'GPU', min_data_in_leaf = 30, depth = 6, od_wait = 75, l2_leaf_reg = 1, iterations = 1500,
                            learning_rate = 0.09, boosting_type = 'Plain', leaf_estimation_method = 'Gradient', bootstrap_type = 'Bernoulli')

ctboost1.fit(train_pool)

pred = ctboost1.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
rmse

# Fit model without specifying any parameters

ctboost2 = CatBoostRegressor()
ctboost2.fit(X = train_pool)

preds = ctboost2.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse

# Repeat prediction of 2020 set

pred_set = df.loc[df.year == 2016, :]

# Get the indices how much a variable increases in 16 years and then get the number it may increase in 2020


cols_to_remove = ['cand_incumbent', 'cand_gender', 'cand_divorcee', 'cand_kids',
                 'cand_senator', 'cand_governor', 'cand_congress', 'cand_veteran', 'cand_vp', 'cand_government', 'cand_ivy', 'cand_age']

pred_set = pred_set.drop(cols_to_remove, 1).reset_index(drop=True)
df3 = df.drop(cols_to_remove,1)

indices = df3.pivot_table(index = ['year', 'FIPS'], values = df3.drop(['year', 'FIPS', 'party', 'state'], 1), aggfunc = 'mean').reset_index().pivot_table(index = 'year', values = df3.drop(['year', 'FIPS', 'party', 'state'], 1), aggfunc = 'sum')
indices = indices.transpose()
indices.columns = ['y2000', 'y2004', 'y2008', 'y2012', 'y2016']
indices['factor'] = (indices['y2016'] / indices['y2000']) ** (1/4)
indices.index.names = ['variable']

dict_factors = dict(zip(indices.index, indices.factor))

del dict_factors['candidatevotes']

# Multiply the variables by trends

for key, value in dict_factors.items():
    pred_set[key] = pred_set[key] * round(value,2)

# import the data about candidates and electoral votes

cands_data = pd.read_excel(os.path.join(file_base, "candidates_2020.xlsx").replace(os.sep, "/"))
electoral_votes = pd.read_excel(os.path.join(file_base, "electoral_votes.xlsx").replace(os.sep, "/"))

# create empty lists

win_party = []
win_candidate = []
win_votes = []
lose_party = []
lose_candidate = []
lose_votes = []
race_black_pct = []
tot_pop_pct = []
income_pc_pct = []
en_college_pct = []
married_pct = []

# Create copy's of dataset for each candidate (10 Democrats, plus Donald Trump)

dset = []

for cand in cands_data.loc[:,'candidate'].tolist():
    cand_set = pd.merge(pred_set.reset_index(drop=True), cands_data.loc[cands_data.candidate == cand], on = 'party', how = 'left')
    dset.append(cand_set)
    
datas = pd.concat(dset).dropna().reset_index(drop=True)
datas['year'] = 2020

# IMPORTANT - how much do you want indices to grow??? 

low_per = 0.02
high_per = 0.02
spread = 4

# Simulation itself

for cand in cands_data.loc[cands_data.party == "democrat", 'candidate'].tolist():
    for a in np.linspace(dict_factors["Enr_college"] - low_per, dict_factors["Enr_college"] + high_per, spread):
        for b in np.linspace(dict_factors["inc_pc"] - low_per, dict_factors["inc_pc"] + high_per, spread):
            for c in np.linspace(dict_factors["Tot_population"] - low_per, dict_factors["Tot_population"] + high_per, spread):
                for d in np.linspace(dict_factors["Race_black"] - low_per, dict_factors["Race_black"] + high_per, spread):
                    for e in np.linspace(dict_factors["Married"] - low_per, dict_factors["Married"] + high_per, spread):
                        
                        datas.loc[:, "Race_black"] = datas.loc[:, "Race_black"] * round(d, 3)
                        datas.loc[:, "Tot_population"] = datas.loc[:, "Tot_population"] * round(c, 3)
                        datas.loc[:, "inc_pc"] = datas.loc[:, "inc_pc"] * round(b, 3)
                        datas.loc[:, "Enr_college"] = datas.loc[:, "Enr_college"] * round(a, 3)
                        datas.loc[:, "Married"] = datas.loc[:, "Married"] * round(e, 3)

                        sett = datas.loc[np.logical_or(datas.candidate == cand, datas.candidate == 'Donald Trump'), :]
                        cands_ages = cands_data.loc[:, ['cand_age', 'candidate']]
                        sett = sett.drop(['candidate', 'candidatevotes'], 1)                        
                        sett['candidatevotes'] = ctboost2.predict(sett)                      
                        sett = pd.merge(sett, cands_ages, on = 'cand_age', how = 'left')                        
                        sett = sett.pivot_table(index = ["state", "party", "candidate"], values = "candidatevotes", aggfunc = "sum").reset_index()                    
                        sett = pd.merge(sett, electoral_votes, on = 'state', how = 'left')

                        maxy = sett.groupby(['state'])['candidatevotes'].transform(max) == sett['candidatevotes']
                        mins = sett.groupby(['state'])['candidatevotes'].transform(min) == sett['candidatevotes']

                        final1 = sett[maxy]
                        final2 = sett[mins]
                        final2.votes = 0
                        final = pd.concat([final1, final2]).reset_index(drop=True)
                        final_breakdown = final.pivot_table(index = ["party", "candidate"], values = "votes", aggfunc = "sum").reset_index()

                        final_max = final_breakdown.loc[final_breakdown.votes == final_breakdown.votes.max() , :]
                        final_min = final_breakdown.loc[final_breakdown.votes == final_breakdown.votes.min() , :]

                        win_party.append(final_max.iloc[0,0])
                        win_candidate.append(final_max.iloc[0,1])
                        win_votes.append(final_max.iloc[0,2])

                        lose_party.append(final_min.iloc[0,0])
                        lose_candidate.append(final_min.iloc[0,1])
                        lose_votes.append(final_min.iloc[0,2])

                        race_black_pct.append(round(d, 3))
                        tot_pop_pct.append(round(c, 3))
                        income_pc_pct.append(round(b, 3))
                        en_college_pct.append(round(a, 3))
                        married_pct.append(round(e, 3))
                        

results = pd.DataFrame(list(zip(win_party, win_candidate, win_votes, lose_party, lose_candidate, lose_votes, 
                                race_black_pct, tot_pop_pct, income_pc_pct, en_college_pct, married_pct)), columns = ["win_party", 
                                "win_candidate", "win_votes", "lose_party", "lose_candidate", "lose_votes", 
                                "race_black_pct", "tot_pop_pct", "income_pc_pct", "en_college_pct", "married_pct"])

results.pivot_table(index = 'win_candidate', aggfunc = 'count').loc[:, 'en_college_pct']

# Check how many electoral votes Donald Trump got in total

r1 = results.loc[results.lose_candidate == 'Donald Trump', ['lose_candidate', 'lose_votes']].pivot_table(index = 'lose_candidate',  aggfunc = 'sum')
r1

# Show all the 'realistic' wins for all the Democratic candidates

results.loc[results.win_votes <= 350, :]#.pivot_table(index = 'win_candidate', columns = 'married_pct', values = 'win_votes', aggfunc = 'count')

