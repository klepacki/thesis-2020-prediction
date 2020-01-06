# -*- coding: utf-8 -*-

# download the data, work on establishing online connection

import pandas as pd
import numpy as np
import os

# missing values

data = pd.read_csv('D:/magisterka_1/dane/countypres_2000-2016.csv', sep=',', header=0 )
data = data.loc[:, data.columns != "version"]

# missing values

missing_summary = data.isnull().sum()

file_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dane/ultimate')

types = ['econ_char', 'social_char', 'population_gender_veteran', 'population_race']
years = ['2000', '2004', '2008', '2012', '2016']

# Define all the operations necessary to be performed on a table

def operations(file1_df, path1_df, labe1l_file_df, path1_labels, year):
    
            # Read the CSV data file
            file1_df = pd.read_csv(path1_df, header = 0, sep = ",", encoding = 'latin')
            
            # Read the CSV labels file
            label1_file_df = pd.read_csv(path1_labels, header = 0, sep = ";", encoding = 'latin')
            
            # Get the columns names and the labels
            to_keep = label1_file_df.iloc[:,0].tolist()
            labels = label1_file_df.iloc[:,1].tolist()
            
            # Drop the first row
            file1_df = file1_df.loc[:, to_keep].drop([0])
            
            # Set the labels
            file1_df.columns = labels
            
            # Add new column with the year
            file1_df['year'] = year
            
            # Change the Id type, so it matches the election results file
            file1_df['Id2'] = file1_df['Id2'].astype('float64')
            
            return file1_df
            
types = ['econ_char', 'social_char', 'population_gender_veteran', 'population_race']
years = ['2000', '2004', '2008', '2012', '2016']
            
# Import the files and perform the operations

file_names = []
files_total = []

for typ in types:
    for year in years:
        
        # Set the name of the data file
        df_name = typ + '_' + year
        file_names.append(df_name)
        
        # Get the data file path right
        path_df = os.path.join(file_base, typ, df_name + ".csv").replace(os.sep,"/")
        file_df = os.path.splitext(path_df)[0]
        file_df = os.path.basename(file_df)

        # Get the label path right
        path_labels = os.path.join(file_base, typ, "label_" + df_name + ".csv").replace(os.sep,"/")
        label_file_df = os.path.splitext(path_labels)[0]
        label_file_df = os.path.basename(label_file_df)
        
        # Perform the operations
        files_total.append(operations(file_df, path_df, label_file_df, path_labels, year))
        
# Create dictionary of all the files

dict_ultimate = dict(zip(file_names, files_total))        

# Population_race - get the df, clean up, change type and check - that one has no missings - can be changed into int

population_race = pd.concat([dict_ultimate["population_race_2016"], dict_ultimate["population_race_2012"], 
                      dict_ultimate["population_race_2008"], dict_ultimate["population_race_2004"],
                      dict_ultimate["population_race_2000"]], axis = 'rows', sort = 'False')

population_race = population_race.drop(['Geography', 'Id'],1).astype('float64')
population_race.info()    
    
# Econ_char - get the df, clean up, change type and go for it

econ_char = pd.concat([dict_ultimate["econ_char_2016"], dict_ultimate["econ_char_2012"], 
                      dict_ultimate["econ_char_2008"], dict_ultimate["econ_char_2004"],
                      dict_ultimate["econ_char_2000"]], axis = 'rows', sort = 'False')

econ_char = econ_char.replace(to_replace=["N","(X)"], value = np.nan).drop(['Geography', 'Id'],1).astype('float64')
econ_char.info()    

# Social_char magic - to idzie do zautomatyzowania!!!!!!

social_char = pd.concat([dict_ultimate["social_char_2016"], dict_ultimate["social_char_2012"], 
                      dict_ultimate["social_char_2008"], dict_ultimate["social_char_2004"],
                      dict_ultimate["social_char_2000"]], axis = 'rows', sort = 'False')


social_char = social_char.replace(to_replace="N", value = np.nan).drop(['Geography', 'Id'], 1).astype('float64').reset_index()

social_char.loc[:, ["Married", "F_married", "M_married"]] = social_char.loc[:, ["Married", "F_married", "M_married"]].fillna(0).astype('int64')
social_char["Married"] = social_char["Married"] + social_char["F_married"] + social_char["M_married"]

social_char["Married"].replace(0, np.nan, inplace = True)  

social_char.loc[:, ["Widowed", "F_widowed", "M_widowed"]] = social_char.loc[:, ["Widowed", "F_widowed", "M_widowed"]].fillna(0).astype('int64')
social_char["Widowed"] = social_char["Widowed"] + social_char["F_widowed"] + social_char["M_widowed"]

social_char["Widowed"].replace(0, np.nan, inplace = True)  

social_char.loc[:, ["Never_married", "F_never_married", "M_never_married"]] = social_char.loc[:, ["Never_married", "F_never_married", "M_never_married"]].fillna(0).astype('int64')
social_char["Never_married"] = social_char["Never_married"] + social_char["F_never_married"] + social_char["M_never_married"]

social_char["Never_married"].replace(0, np.nan, inplace = True)  

social_char.loc[:, ["Divorced", "F_divorced", "M_divorced"]] = social_char.loc[:, ["Divorced", "F_divorced", "M_divorced"]].fillna(0).astype('int64')
social_char["Divorced"] = social_char["Divorced"] + social_char["F_divorced"] + social_char["M_divorced"]

social_char["Divorced"].replace(0, np.nan, inplace = True)

social_char.loc[:, ["Separated", "F_separated", "M_separated"]] = social_char.loc[:, ["Separated", "F_separated", "M_separated"]].fillna(0).astype('int64')
social_char["Separated"] = social_char["Separated"] + social_char["F_separated"] + social_char["M_separated"]

social_char["Separated"].replace(0, np.nan, inplace = True)

social_char = social_char.drop(["F_married", "M_married", "F_divorced", "M_divorced", "F_separated", "M_separated", "F_widowed", "M_widowed",
                  "F_never_married", "M_never_married", "index", "Tot_population"], 1)

social_char.info()

# Population_gender_veteran - standard stuff
 
population_gender_veteran = pd.concat([dict_ultimate["population_gender_veteran_2016"], dict_ultimate["population_gender_veteran_2012"], 
                      dict_ultimate["population_gender_veteran_2008"], dict_ultimate["population_gender_veteran_2004"],
                      dict_ultimate["population_gender_veteran_2000"]], axis = 'rows', sort = 'False')

population_gender_veteran = population_gender_veteran.replace(to_replace="N", value = np.nan).drop(['Id', 'Geography'], 1).astype('float64').reset_index()
        
population_gender_veteran.loc[:, ["F_veterans", "F_veterans_18to34", "F_veterans_18to64", "F_veterans_35to54", "F_veterans_55to64",
                    "F_veterans_65plus", "F_veterans_65to74", "F_veterans_75plus"]] = population_gender_veteran.loc[:, ["F_veterans", "F_veterans_18to34", "F_veterans_18to64", "F_veterans_35to54", "F_veterans_55to64",
                    "F_veterans_65plus", "F_veterans_65to74", "F_veterans_75plus"]].fillna(0).astype('int64')

population_gender_veteran["F_veterans"] = population_gender_veteran["F_veterans"] + population_gender_veteran["F_veterans_18to34"] + population_gender_veteran["F_veterans_18to64"] + population_gender_veteran["F_veterans_35to54"] + population_gender_veteran["F_veterans_55to64"] + population_gender_veteran["F_veterans_65plus"] + population_gender_veteran["F_veterans_65to74"] + population_gender_veteran["F_veterans_75plus"]
                                    
population_gender_veteran["F_veterans"].replace(0, np.nan, inplace = True)

population_gender_veteran.loc[:, ["M_veterans", "M_veterans_18to34", "M_veterans_18to64", "M_veterans_35to54", "M_veterans_55to64",
                    "M_veterans_65plus", "M_veterans_65to74", "M_veterans_75plus"]] = population_gender_veteran.loc[:, ["M_veterans", "M_veterans_18to34", "M_veterans_18to64", "M_veterans_35to54", "M_veterans_55to64",
                    "M_veterans_65plus", "M_veterans_65to74", "M_veterans_75plus"]].fillna(0).astype('int64')

population_gender_veteran["M_veterans"] = population_gender_veteran["M_veterans"] + population_gender_veteran["M_veterans_18to34"] + population_gender_veteran["M_veterans_18to64"] + population_gender_veteran["M_veterans_35to54"] + population_gender_veteran["M_veterans_55to64"] + population_gender_veteran["M_veterans_65plus"] + population_gender_veteran["M_veterans_65to74"] + population_gender_veteran["M_veterans_75plus"]
                                    
population_gender_veteran["M_veterans"].replace(0, np.nan, inplace = True)

population_gender_veteran = population_gender_veteran.drop(["F_veterans_18to34", "F_veterans_18to64", "F_veterans_35to54", "F_veterans_55to64",
                    "F_veterans_65plus", "F_veterans_65to74", "F_veterans_75plus", "M_veterans_18to34", "M_veterans_18to64", "M_veterans_35to54", "M_veterans_55to64",
                    "M_veterans_65plus", "M_veterans_65to74", "M_veterans_75plus", "Tot_veterans", "index"], 1)

population_gender_veteran.info()
    
# Eliminate all the counties without FIPS and change the type of the year

data = data.dropna(subset = ["FIPS"])
data["year"] = data["year"].astype('float64')


# Join the data together

join_key = ["FIPS", "year"]
tables_indices = ["Id2", "year"]
tables = [population_gender_veteran, population_race, econ_char, social_char]

data = data.set_index(join_key)

for boy in tables:
    data = data.join(boy.set_index(tables_indices), on=join_key)
    
data = data.loc[np.logical_or(data["party"] == "democrat", data["party"] == "republican"), :]
    
# Get the candidates file and join with data

candidates_2000_2016 = pd.read_excel(os.path.join(file_base, "candidates_2000_2016.xlsx").replace(os.sep, '/'))

candidates_2000_2016['year'] = candidates_2000_2016['year'].astype('float64')

data = data.reset_index().set_index(['year', 'candidate']).join(candidates_2000_2016.drop('party', 1).set_index(['year', 'candidate']).sort_index(), on = ['year', 'candidate']).reset_index()

data.to_csv(os.path.join(file_base, "dane_ultimate1.csv").replace(os.sep,"/"), sep = ",", encoding = 'utf-8')