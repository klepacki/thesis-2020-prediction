#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

import os

import talos as ta
from talos.utils import lr_normalizer
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

file_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dane/ultimate')

df = pd.read_csv(os.path.join(file_base, 'dane_imp_finale1.csv').replace(os.sep, '/'), sep = ',')
df = df.drop(df.columns[0], axis=1)
df = df.drop(['state_po', 'county', 'candidate', 'cand_mayor', 'cand_married'],1)

# Graph plot losses

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

# Label Encoder    
    
to_encode = ['party', 'state']
labs = LabelEncoder()

for encoder in to_encode:
    df[encoder] = labs.fit_transform(df[encoder])  

# Splitting data set into test and training for both scaled and non-scaled datasets - scaled

cand_votes = df.loc[:, 'candidatevotes']
features = df.drop(['candidatevotes'], 1)

X_train, X_test, y_train, y_test = train_test_split(features, cand_votes, random_state = 31, test_size = 0.2)

# Standardizing the data

mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)

X_train = (X_train-mean)/std
X_test = (X_test-mean)/std

X_test = X_test.reset_index().drop('index', 1)
X_train = X_train.reset_index().drop('index', 1)
y_train = y_train.reset_index().drop('index', 1)
y_test = y_test.reset_index().drop('index', 1)

# Set the parameters space

p = {'dropout': [0.25, 0.5, 0.75],
         'epochs': [500, 1000, 1500, 2000],
         'batch_size': [256, 512, 1024],
         'loss_func': ['mse'],
         'patience': [25, 50, 75],
         'lr': [0.1, 0.05, 0.01, 0.005, 0.001],
         'optimizer': [Adam, Nadam, RMSprop],
         'activation': ['relu', 'selu'],
         'first_neuron': [32, 64, 128],
         'second_layer': [8, 16, 32],
        }

# Define the network

def the_network(X_train, y_train, X_test, y_test, params):

    n_network = Sequential()
    n_network.add(Dense(params['first_neuron'], input_dim = X_train.shape[1], activation = 'linear'))
    n_network.add(BatchNormalization())
    n_network.add(Dense(params['second_layer'], activation = params['activation']))
    n_network.add(Dropout(params['dropout']))
    n_network.add(Dense(1,  activation = params['activation']))

    # Compiling the network

    n_network.compile(optimizer = params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), 
                      loss = params['loss_func'], 
                      metrics = ['mse'])

    # Adding callback - early stopping

    cb = EarlyStopping(monitor = 'mse', patience = params['patience'])

    # Fitting the network and creating fitting plots

    history = n_network.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = params['epochs'], verbose = 0, batch_size = params['batch_size'], callbacks = [cb])

    return history, n_network
    
    plot_loss(history.history['loss'], history.history['val_loss'])

# Run randomized search using Talos library with 0.01 fraction limit

scan_object = ta.Scan(X_train, y_train, params = p, model = the_network, fraction_limit = 0.01, experiment_name = 'boy_4_12')
analyze_object = ta.Analyze(scan_object)

# Plot 'training' bars

analyze_object.plot_bars('batch_size', 'mse', 'lr', 'first_neuron')

# Run reporting on training log - to be changed accordingly

from talos import Reporting

r = Reporting(os.path.join(r'C:\Users\Szymek\Documents\magisterka_1\boy_4_12', '120419221138.csv').replace(os.sep, '/'))


# Find the lowest score

r.low(metric = 'loss')


# Find the round number with the lowest score

r.data.loss.idxmin()


# Plot the loss function across epochs

r.plot_line(metric = 'loss')


# Plot correlation matrix

analyze_object.plot_corr(metric = 'loss', exclude = ['val_loss', 'mse', 'val_mse'])


# Run the neural network on best parameters - selected manually - to be automated

n_network = Sequential()

n_network.add(Dense(32, input_dim = X_train.shape[1], activation = 'linear'))
n_network.add(BatchNormalization())
n_network.add(Dense(32, activation = 'relu'))
n_network.add(Dropout(0.25))
n_network.add(Dense(1,  activation = 'relu'))


# Compiling the network

n_network.compile(optimizer = Adam(lr=lr_normalizer(0.1, Adam)), 
                      loss = 'mse', 
                      metrics = ['mse'])


# Adding callback - early stopping

cb = EarlyStopping(monitor = 'mse', patience = 75)


# Fitting the network and creating fitting plots

history = n_network.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 1000, verbose = 1, batch_size = 512, callbacks = [cb])

plot_loss(history.history['loss'], history.history['val_loss'])


# DNN - prediction

preds = n_network.predict(X_test)


# Calculating RMSE of DNN

rmse = np.sqrt(mean_squared_error(y_test, preds))

rmse


# Cut the data for 2016

pred_set = df.loc[df.year == 2016, :]


# Get the indices how much a variable increases in 16 years and then get the number it may increase in 2020

cols_to_remove = ['cand_incumbent', 'cand_gender', 'cand_divorcee', 'cand_kids',
                 'cand_senator', 'cand_governor', 'cand_congress', 'cand_veteran', 'cand_vp', 'cand_government', 'cand_ivy', 'cand_age']

pred_set = pred_set.drop(cols_to_remove, 1).reset_index(drop=True)


# Get the values of increase ratio to create dataset with values for 2020 and put it into dictionary

df3 = df.drop(cols_to_remove,1)

indices = df3.pivot_table(index = ['year', 'FIPS'], values = df3.drop(['year', 'FIPS', 'party', 'state'], 1), aggfunc = 'mean').reset_index().pivot_table(index = 'year', values = df3.drop(['year', 'FIPS', 'party', 'state'], 1), aggfunc = 'sum')
indices = indices.transpose()
indices.columns = ['y2000', 'y2004', 'y2008', 'y2012', 'y2016']
indices['factor'] = (indices['y2016'] / indices['y2000']) ** (1/4)
indices.index.names = ['variable']

dict_factors = dict(zip(indices.index, indices.factor))

del dict_factors['candidatevotes']


# Mutliply dataset by values of dictionary

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

                        to_encode = ['party', 'state']
                        labs = LabelEncoder()

                        for encoder in to_encode:
                            sett[encoder] = labs.fit_transform(sett[encoder])

                        sett = sett.drop(['candidate', 'candidatevotes'], 1)
                        sett = (sett - mean)/std
                        sett['candidatevotes'] = pd.Series(np.reshape(n_network.predict(sett), -1))
                        sett['candidate'] = sett['party'].apply(lambda x: cand if x < 0 else 'Donald Trump')
                        sett['party'] = sett['party'].apply(lambda x: 'republican' if x > 0 else 'democrat')

                        state_help = pd.merge(pd.DataFrame(sett['state'].unique(), columns = ['state']), electoral_votes, left_index = True, right_index = True)

                        sett = pd.merge(sett, state_help, left_on = 'state', right_on = 'state_x')
                        sett = sett.drop(['state_x', 'votes', 'state'], 1).rename(columns = {'state_y': 'state'})
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


# Check how many times a particular candidate won with regard to 'en_college_pct' variable - can be done for others as well

results.pivot_table(index = 'win_candidate', aggfunc = 'count').loc[:, 'en_college_pct']


# Check the votes of other Democratic candidates except for Biden

r1 = results.loc[(results.lose_candidate != 'Donald Trump') & (results.lose_candidate != 'Joe Biden'), ['lose_candidate', 'lose_votes']].pivot_table(index = 'lose_candidate',  aggfunc = 'count')
r1['lose_votes'] = 0


# Check Biden vs. Trump results 

r1 = results.loc[((results.win_candidate == 'Donald Trump') & (results.lose_candidate == 'Joe Biden')) | ((results.win_candidate == 'Joe Biden') & (results.lose_candidate == 'Donald Trump')) , :]


# Check results for 'married_pct' variable - it can be done similarily for other ones

r1.loc[r1.win_votes <= 350, :].pivot_table(index = 'win_candidate', columns = 'married_pct', values = 'win_votes', aggfunc = 'count')

