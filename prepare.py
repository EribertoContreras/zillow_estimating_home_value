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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import warnings
warnings.filterwarnings('ignore')
#removing outliers to get a better look at the data
def remove_outliers(threshold, quant_cols, df):
    z = np.abs((stats.zscore(df[quant_cols])))
    df_without_outliers=  df[(z < threshold).all(axis=1)]
    print(df.shape)
    print(df_without_outliers.shape)
    non_quants = ['yearbuilt', 'fips','propertylandusedesc']
    quants = df.drop(columns=non_quants).columns
    df = remove_outliers(3.5, quants, df) 
    return df_without_outliers

def clean_zillow_data(df):
    """
    Takes in zillow Dataframe from the get_zillow_data function.
    Arguments: drops unnecessary columns, 0 value columns, duplicates,
    and converts select columns from float to int.
    Returns cleaned data.
    """ 
    # remove empty entries stored as whitespace, convert to nulls
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # drop null rows
    #df = df.dropna()
    #drop any duplicate rows
    df = df.drop_duplicates(keep='first')
    #df['column name'] = df['column name']. replace(['old value'],'new value')
    # df['fips'] = df['fips'].replace(6037.0, 'Los Angeles,CA')
    # df['fips'] = df['fips'].replace(6059.0, 'Orange,CA')
    # df['fips'] = df['fips'].replace(6111.0, 'Ventura,CA')
    # ventura = df[df.fips == 'Ventura,CA']
    # la = df[df.fips == 'Los Angeles,CA']
    # oc = df[df.fips == 'Orange,CA']
    # remove homes with 0 BR/BD or SQ FT from the final df, dropping rows that have zip code = 0
    df = df[(df.bedroomcnt != 0) & (df.bathroomcnt != 0) & (df.calculatedfinishedsquarefeet >= 500)&(df.bedroomcnt < 6)&(df.bedroomcnt > 1)&(df.regionidzip != 0.0)&(df.bathroomcnt < 4)
           & (df.yearbuilt > 1899)&(df.calculatedfinishedsquarefeet < 3050)&(df.taxvaluedollarcnt < 1000000)&(df.bathroomcnt > 1)&(df.yearbuilt > 1919)&(df.yearbuilt <= 2015)]
    
    # dropping collumns that will only get us more comfused in the exploration process.
    df = df.drop(columns=["parcelid",
                 "id",
                 "airconditioningtypeid",
                 "architecturalstyletypeid",
                 "basementsqft",
                 "buildingclasstypeid",
                 "buildingqualitytypeid",
                 "calculatedbathnbr",
                 "decktypeid",
                 'garagecarcnt',
                 "finishedfloor1squarefeet",
                 'finishedsquarefeet12',
                 'finishedsquarefeet13',
                 'finishedsquarefeet15',
                 'finishedsquarefeet50',
                 'finishedsquarefeet6',
                 'fireplacecnt',
                 'fullbathcnt',
                 'heatingorsystemtypeid',
                 'garagetotalsqft',
                 'hashottuborspa',
                 'lotsizesquarefeet',
                 'poolcnt',
                 'regionidcity',
                 'poolsizesum',
                 'pooltypeid10',
                 'pooltypeid2',
                 'pooltypeid7',
                 'landtaxvaluedollarcnt',
                 'structuretaxvaluedollarcnt',
                 'taxamount',
                 'propertycountylandusecode',
                 'propertylandusetypeid',
                 'propertyzoningdesc',
                 'rawcensustractandblock',
                 'roomcnt',
                 'logerror',
                 'storytypeid',
                 'threequarterbathnbr',
                 'typeconstructiontypeid',
                 'unitcnt',
                 'yardbuildingsqft17',
                 'yardbuildingsqft26',
                 'numberofstories',
                 'fireplaceflag',
                 'structuretaxvaluedollarcnt',
                 'assessmentyear',
                 'taxdelinquencyflag',
                 'taxdelinquencyyear',
                 'censustractandblock',
                 'propertylandusedesc',
                 'id',
                 'transactiondate','regionidneighborhood','id.1'])
    # filling iin NaN with 0, to fill in values that have no info like garages, heatingsystems and A/C 
    df = df.fillna(0)
    return df 

def split_zillow_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on fips.
    return train, validate, test DataFrames.
    '''
        #splits df into train_validate and test using train_test_split() stratifying on fips to get an even mix of each fips
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
        # splits train_validate into train and validate using train_test_split() stratifying on fips to get an even mix of each fips
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test

    # copy and paste this on notebook to summon split data train .train, validate, test = prepare.split_zillow_data(df)

# scaling my data will allow me to visualize and explore my data easier, makinng the visuals alot better to comprehend
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bathroomcnt', 'bedroomcnt',
                 'latitude', 'longitude', 'regionidcounty', 'regionidzip', 'yearbuilt',
                 'calculatedfinishedsquarefeet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

        #copy and paste this to your jupyter notebook so you can get test,train,validate scaled (scaler, train_scaled, validate_scaled, test_scaled = prepare.scale_data(train, validate, test, return_scaler=True))

#______________________________________________________________________________________________________________________________________________________________________________________________


def plot_variable_pairs(train):
    columns = ['calculatedfinishedsquarefeet','bathroomcnt','bedroomcnt','yearbuilt']
    for col in columns:
        sns.lmplot(data = train.sample(10000), x = col, y='taxvaluedollarcnt')#,hue='fips',col='fips', line_kws= {'color': 'red'},data=train.sample(1000))
    


def plot_categorical_and_continuous_vars(train):
    columns = ['calculatedfinishedsquarefeet','bathroomcnt','bedroomcnt']
    for col in columns:
        sns.set(rc={"figure.figsize":(15, 6)})
        fig, axes = plt.subplots(1,3)
        
        sns.boxplot(x='fips', y=col, data=train.sample(1000),ax = axes[0])
        sns.violinplot(x='fips', y= col, data=train.sample(1000),ax = axes[1]) 
        sns.swarmplot(x='fips', y= col, data=train.sample(1000),ax = axes[2])
        
        plt.show
        