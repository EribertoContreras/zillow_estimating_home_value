# clean_zillow_data(df) function is created to cut and clean data that could affect future work. 
# wrangle_zillow() is a function created for the purpose of combining both get_zillow_data and clean_zillow_data.

from cgi import test
from lib2to3.pgen2.pgen import DFAState
from lib2to3.refactor import get_all_fix_names
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import env
from pydataset import data
import scipy
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def clean_zillow_data(df):
    """
    Takes in zillow Dataframe from the get_zillow_data function.
    Arguments: drops unnecessary columns, 0 value columns, duplicates,
    and converts select columns from float to int.
    Returns cleaned data.
    """
    # remove empty entries stored as whitespace, convert to nulls
    #df = df.replace(r'^\s*$', np.nan, regex=True)
    # drop null rows
    #df = df.dropna()
    #drop any duplicate rows
    df = df.drop_duplicates(keep='first')
    # remove homes with 0 BR/BD or SQ FT from the final df
    df = df[(df.bedroomcnt != 0) & (df.bathroomcnt != 0) &
    (df.calculatedfinishedsquarefeet >= 69)]
    #df['column name'] = df['column name']. replace(['old value'],'new value')
    df['fips'] = df['fips'].replace(6037.0, 'Los Angeles,CA')
    df['fips'] = df['fips'].replace(6059.0, 'Orange,CA')
    df['fips'] = df['fips'].replace(6111.0, 'Ventura,CA')
    return df 


# def split_zillow_data(df):
#     '''
#     take in a DataFrame and return train, validate, and test DataFrames; stratify on fips.
#     return train, validate, test DataFrames.
#     '''
#         #splits df into train_validate and test using train_test_split() stratifying on fips to get an even mix of each fips
#     train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
#         # splits train_validate into train and validate using train_test_split() stratifying on fips to get an even mix of each fips
#     train, validate = train_test_split(train_validate, 
#                                        test_size=.3, 
#                                        random_state=123)
#     return train, validate, test

# train, validate, test = split_zillow_data(df)

# columns = ['yearbuilt','calculatedfinishedsquarefeet','taxamount','bathroomcnt','bedroomcnt']

# def plot_variable_pairs():
#     for col in columns:
#         sns.lmplot(x = col, y='taxvaluedollarcnt', col = 'fips',hue='fips',
#                line_kws= {'color': 'red'},data=train.sample(1000))
    
# plot_variable_pairs()

# columns = ['yearbuilt','calculatedfinishedsquarefeet','taxamount','bathroomcnt','bedroomcnt']
# def plot_categorical_and_continuous_vars():
#     for col in columns:
#         sns.set(rc={"figure.figsize":(15, 6)})
#         fig, axes = plt.subplots(1,3)
        
#         sns.boxplot(x='fips', y=col, data=train.sample(1000),ax = axes[0])
#         sns.violinplot(x='fips', y= col, data=train.sample(1000),ax = axes[1]) 
#         sns.swarmplot(x='fips', y= col, data=train.sample(1000),ax = axes[2])
        
#         plt.show
        
# plot_categorical_and_continuous_vars()