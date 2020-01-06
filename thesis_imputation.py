# -*- coding: utf-8 -*-


import pandas as pd 
import os
import numpy as np
import missingno as msno

import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor


file_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dane/ultimate')

# Get the data

dane_ssd = pd.read_csv(os.path.join(file_base, "dane_ultimate1.csv").replace(os.sep, "/"), header = 0, sep = ',', encoding = 'utf-8', index_col = [0])

# Missing analysis

dane_missing = dane_ssd.columns[dane_ssd.isnull().any()].tolist()

msno.matrix(dane_ssd[dane_missing]).get_figure().savefig(os.path.join(file_base, "outputs/missing_pattern.png"))

msno.bar(dane_ssd[dane_missing], color="blue", log=True, figsize=(30,18)).get_figure().savefig(os.path.join(file_base, "missing_count.png"))

msno.heatmap(dane_ssd[dane_missing], figsize=(20,20)).get_figure().savefig(os.path.join(file_base, "outputs/missing_heatmap.png"))

# Display missing percentage

missings = dane_ssd.isna().sum().reset_index()

missings.columns = ['variable', 'percentage']

missings['percentage'] = missings['percentage'] * 100/len(dane_ssd)

pd.set_option('display.max_rows', 73)

missings[missings['percentage'] > 0]


# Impute missing data

dane_not_missing = dane_ssd.columns[dane_ssd.notnull().all()].tolist()
columns_out = ['candidate', 'party', 'state', 'state_po', 'county', 'office']

N_SPLITS = 5

estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15)
]

def checker(dataset):

    checker1 = pd.merge(dataset.pivot_table(index = ['year','FIPS'], values = 'Tot_population', aggfunc = 'mean').pivot_table(index='year', values = 'Tot_population', aggfunc = 'sum').reset_index(),
          dataset.pivot_table(index = 'year', values = 'candidatevotes', aggfunc = 'sum').reset_index())

    checker1['turnout_pred'] = checker1['candidatevotes'] * 100 / checker1['Tot_population']
    checker1 = pd.merge(checker1, pd.DataFrame(data=turnouts_r))
    checker_coeff1 = np.mean((checker1['turnout_pred'] - checker1['turnout'])**2)
    return checker_coeff1

turnouts_r ={'year': [2000.0, 2004.0, 2008.0, 2012.0, 2016.0],
             'turnout': [51.2, 56.7, 58.2, 54.9, 55.7]}

coeffs = []

for estim in estimators:
    Imputer = IterativeImputer(estimator = estim)
    
    dane_ssd_imp1 = pd.DataFrame(data = Imputer.fit_transform(dane_ssd.drop(columns_out, 1)), columns = dane_ssd.drop(columns_out, 1).columns.tolist())
    
    checker(dane_ssd_imp1)

    coeffs.append(checker_coeff)
    
coeffs_comp = pd.DataFrame({'Estimators':['Bayesian Ridge', 'Decision Tree', 'Extra Trees', 'KNNeighbors'],
                            'Turnout Coeff.':coeffs})
    
coeffs_comp

# Best imputer chosen - KNNeighborsRegressor

# TO BE AUTOMATED

Imputer = IterativeImputer(estimator = KNeighborsRegressor(n_neighbors = 15))

dane_ssd_imp = pd.concat([pd.DataFrame(data = Imputer.fit_transform(dane_ssd.drop(columns_out, 1)), columns = dane_ssd.drop(columns_out, 1).columns.tolist()), 
                          dane_ssd[columns_out]], axis=1)

dane_ssd_imp.to_csv(os.path.join(file_base, "dane_ssd_po_imputacji.csv").replace(os.sep, "/"), index = None)

# Voters turnout - compare

columns_out = ['candidate', 'party', 'state', 'state_po', 'county', 'cand_age', 'cand_incumbent', 'cand_gender', 'cand_married', 
               'cand_divorcee', 'cand_kids', 'cand_senator', 'cand_governor', 'cand_congress',
               'cand_veteran', 'cand_vp', 'cand_government', 'cand_mayor', 'cand_ivy', 'totalvotes', 'candidatevotes']


dane_ssd_imp = dane_ssd_imp.drop('office',1)

dane_ssd_imp_impute = dane_ssd_imp.drop(columns_out, 1)
dane_ssd_imp_remain = dane_ssd_imp[columns_out + ['year','FIPS']]
dane_ssd_imp_remain.loc[:, ['year', 'FIPS']] = dane_ssd_imp_remain.loc[:, ['year', 'FIPS']].astype('int64')
dane_ssd_imp_remain = dane_ssd_imp_remain.set_index(['year', 'FIPS'])


dane_turnout1 = dane_ssd_imp_impute.pivot_table(index = ['year', 'FIPS'], values = dane_ssd_imp_impute.drop(['year', 'FIPS'], 1), aggfunc = 'mean').pivot_table(index = 'year', values = dane_ssd_imp_impute.drop(['year', 'FIPS'], 1), aggfunc = 'sum')


dane_turnout_ult = pd.concat([dane_ssd_imp.pivot_table(index = "year", values = 'candidatevotes', aggfunc = 'sum'), dane_turnout1], axis = 1).reset_index().astype('int64').set_index('year')
dane_turnout_ult = dane_turnout_ult.astype('int64').drop('candidatevotes', 1).transpose().reset_index()

kolumny = ['index', 'y2000', 'y2004', 'y2008', 'y2012', 'y2016']

dane_turnout_ult.columns = kolumny

dane_turnout_ult['y2016_00'] = dane_turnout_ult['y2016'] / dane_turnout_ult['y2000']

dane_turnout_ult['y2004_00'] = dane_turnout_ult['y2004'] / dane_turnout_ult['y2000']

dane_turnout_ult['y2008_04'] = dane_turnout_ult['y2008'] / dane_turnout_ult['y2004']

dane_turnout_ult['y2012_08'] = dane_turnout_ult['y2012'] / dane_turnout_ult['y2008']

dane_turnout_ult['y2016_12'] = dane_turnout_ult['y2016'] / dane_turnout_ult['y2012']

dane_turnout_ult['i2004'] = dane_turnout_ult['y2016_00'] ** (1/4)



dane_2000 = dane_ssd_imp_impute.loc[dane_ssd_imp.year == 2000, :].astype('int64').set_index(['year', 'FIPS']).sort_index(axis=1).reset_index()
dane_2004 = dane_ssd_imp_impute.loc[dane_ssd_imp.year == 2004, :].astype('int64').set_index(['year', 'FIPS']).sort_index(axis=1).reset_index()
dane_2008 = dane_ssd_imp_impute.loc[dane_ssd_imp.year == 2008, :].astype('int64').set_index(['year', 'FIPS']).sort_index(axis=1).reset_index()
dane_2012 = dane_ssd_imp_impute.loc[dane_ssd_imp.year == 2012, :].astype('int64').set_index(['year', 'FIPS']).sort_index(axis=1).reset_index()
dane_2016 = dane_ssd_imp_impute.loc[dane_ssd_imp.year == 2016, :].astype('int64').set_index(['year', 'FIPS']).sort_index(axis=1).reset_index()


for num, (val1, val2) in enumerate(zip(dane_turnout_ult.loc[:, 'i2004'].tolist(), dane_turnout_ult.loc[:, 'y2004_00'].tolist())):
    dane_2004.iloc[:, num + 2] = dane_2004.iloc[:, num + 2] * round(val1,2) / round(val2,2)
   
for num, val in enumerate(dane_turnout_ult.loc[:, 'y2008_04'].tolist()):
    dane_2008.iloc[:, num + 2] = dane_2004.iloc[:, num + 2] * round(val,2)

for num, val in enumerate(dane_turnout_ult.loc[:, 'y2012_08'].tolist()):
    dane_2012.iloc[:, num + 2] = dane_2008.iloc[:, num + 2] * round(val,2)

dane_imp_finale = pd.concat([pd.concat([dane_2000, dane_2004, dane_2008, dane_2012, dane_2016]).astype('int64').reset_index(), dane_ssd_imp[columns_out]], axis = 1)

checker(dane_imp_finale)

cols = ['cand_age', 'cand_incumbent', 'cand_gender', 'cand_married', 
               'cand_divorcee', 'cand_kids', 'cand_senator', 'cand_governor', 'cand_congress',
               'cand_veteran', 'cand_vp', 'cand_government', 'cand_mayor', 'cand_ivy', 'totalvotes', 'candidatevotes']

dane_imp_finale[cols] = dane_imp_finale[cols].astype('int64')

dane_imp_finale = dane_imp_finale.drop('F_veterans',1)

dane_imp_finale.to_csv(os.path.join(file_base, "dane_imp_finale1.csv").replace(os.sep, "/"), index = None)