Scenario
You are a junior data scientist on the Zillow data science team and recieve the following email in your inbox:

We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.

We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

One last thing, Zach lost the email that told us where these properties were located. Ugh, Zach :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

-- The Zillow Data Science Team

Key Takeaways
Major factors determining property price are:
calculatedfinishedsquarefeet
bathroomcnt
bathroomcnt
A LinnearRegression model is the top predictor for this sample of data.
Goals
Construct an ML Regression model that predict propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

Find the key drivers of property value for single family properties.

Use the 2017 Zillow data set.

Make recommendations on what works or doesn't work in prediction these homes' values.

create a predictive model that is better then the baseline model so we can better predict prices for the year

Hypothesis
there is no linear correlation between the calculatedfinishedsquarefeet of a home and taxvaluedollarcnt

there is no linear correlation between the bathroomcnt of a home and taxvaluedollarcnt

there is no linear correlation between the bedroomcnt of a home and taxvaluedollarcnt

there is no linear correlation between the regionidzip of a home and taxvaluedollarcnt

Plan
Acquire Data
Clean and Prep the Data
Explore the features of the Zillow data
Evaluate Data
Test Data
Acquiring the Data
Zillow Data was obtained from our friends at CodeUp. Information on how to acquire the data can be found on this link, please ensure that you have the correct credentials in order to access the data and that they are saved in an env.py file.




    
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
                 'transactiondate',
                 'regionidneighborhood',
                 'id.1'])


We started with 52442 rows of data & 63 columns.
Homes near the coast line & in the mountain side seem to range over $900k.
there seems to be a lot of na values
will be removing everything over 1000
will be keeping calculatefinishedsquarefeet and removing anything else that includes square feet.
Will be keeping fips and removing anything else that has to do with ZIP Code and region(fips is the county number)
will be removing lot sizes, mainly to focus on our calculatefinishedsquarefeet
will be removing garages to focus more on the data that is over %50 complete
getting rid of any id types so that we can have valid numerical data
We will be focusing on bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, fips, latitude, longitude, regionidcounty, regionidzip, yearbuilt,taxvaluedollarcnt, because those seem to be the top criticals factors that influance home values. Factors that influance home price

outliers removed
bedroom count 0
bathroom count 0
calculatedfinishedsquarefeet less than 500
bedroomcnt greater than 6
bedroomcnt count less than 1
regionzip equal to 0
bathroomcnt greater than 4
yearbuilt less than 1899
calculatedfinishedsquarefeet greater than 3050
taxvaluedollarcnt greater than 1,700,000
bathroomcnt less than or equal 1
yearbuilt less than 1919
yearbuilt greater than or equal to 2015

there is a linear correlation between the bedroomcnt of a home and taxvaluedollarcnt
there is a linear correlation between the bathroomcnt of a home and taxvaluedollarcnt
there is a linear correlation between the calculatedfinishedsquarefeet of a home and taxvaluedollarcnt




Conclusion
Major drivers of the house prices:
calculatedfinishedsquarefeet,bedroomcnt, & bathroomcnt sit at the top as far as drivers of churn.It is a good idea to coniseder that the counties of ventura, Los Angeles, and Orange all have a wide range of homes that can be estimated up to 3 million dollars.The more bedrooms, bathrooms and living squarefeet a home has, the bigger the impact it will have on its price.
Recommendations for the future:
My recomendation to the zillow team, for future referance and for more accurate predictions; If I can be given more time to explore the data further so that we may be able to improve our predictions. By identifying more factors that drive home prices, we will increase our likelyhood that our predcition model will be more accurate. 