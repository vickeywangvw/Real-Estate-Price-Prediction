#!/usr/bin/env python
# coding: utf-8

# # **Load Data**

# ## Set up Data Path

# In[758]:


## Iniitialization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATA_PATH = os.getcwd() + '/data/'


# ## Iniitialization
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# 
# DATA_PATH = os.getcwd() + '/data/'

# In[759]:


df = pd.read_csv(os.getcwd() + 'final.sales.csv',skipinitialspace=True,low_memory=False)
print('number of rows:'+str(df.shape[0])+'\n'+'number of columns:'+str(df.shape[1]))


# In[760]:


df.info()
df.shape


# In[761]:


df.head(5)


# # **Data Cleaning**

# ## Edit Column Name

# In[762]:


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# ## Reformat Part of Object Columns Contents into Numeric 

# In[763]:


df['LAND SQUARE FEET']=df['LAND SQUARE FEET'].apply(lambda x: x.replace(',',''))
df['LAND SQUARE FEET']=df['LAND SQUARE FEET'].apply(lambda x: x.replace('?',''))
df['GROSS SQUARE FEET']=df['GROSS SQUARE FEET'].apply(lambda x: x.replace(',',''))

df['RESIDENTIAL UNITS']=df['RESIDENTIAL UNITS'].apply(lambda x: x.replace(',',''))
df['COMMERCIAL UNITS']=df['COMMERCIAL UNITS'].apply(lambda x: x.replace(',',''))
df['TOTAL UNITS']=df['TOTAL UNITS'].apply(lambda x: x.replace(',',''))

df['SALE PRICE']=df['SALE PRICE'].apply(lambda x: x.replace('$',''))
df['SALE PRICE']=df['SALE PRICE'].apply(lambda x: x.replace(',',''))
df['SALE PRICE']=df['SALE PRICE'].apply(lambda x: x.replace('-',''))
df['SALE PRICE']=df['SALE PRICE'].apply(lambda x: x.replace('?',''))
df['SALE PRICE'] = df['SALE PRICE'].str.replace(' ', '').str.replace(str(chr(8377)), '').copy()
df['SALE PRICE'].replace('', np.nan, inplace=True)


# ## Edit Column Type

# In[764]:


df['LAND SQUARE FEET']=df['LAND SQUARE FEET'].astype(int)
df['GROSS SQUARE FEET']=df['GROSS SQUARE FEET'].astype(int)
df['SALE PRICE']=df['SALE PRICE'].astype(float)

df['RESIDENTIAL UNITS']=df['RESIDENTIAL UNITS'].astype(float)
df['COMMERCIAL UNITS']=df['COMMERCIAL UNITS'].astype(float)
df['TOTAL UNITS']=df['TOTAL UNITS'].astype(float)
df['YEAR BUILT']=df['YEAR BUILT'].astype(int)
df['TAX CLASS AT TIME OF SALE']=df['TAX CLASS AT TIME OF SALE'].astype(int)
df['BOROUGH']=df['BOROUGH'].astype('object')
df['SALE DATE']=df['SALE DATE'].astype('datetime64[ns]')


# In[765]:


df.info()


# ## Remove Na/NaN and Unnecessary Columns

# In[766]:


df.drop(['EASE-MENT','ADDRESS','APARTMENT NUMBER','BUILDING CLASS AT PRESENT','BUILDING CLASS AT TIME OF SALE','NEIGHBORHOOD'], 
        axis = 1, inplace=True)


# In[767]:


df.isna().sum()


# In[768]:


df = df.dropna()


# In[769]:


df.isna().sum()


# In[770]:


df.fillna('NA', inplace=True)
df.isna().sum()


# ## Filter Data Values

# ### Sale Price

# In[771]:


# remove data records with zip code < 100000
df = df[df['SALE PRICE']!=0]
df['SALE PRICE'].describe()


# In[772]:


# Plot the distribution of SALE PRICE
plt.boxplot(df['SALE PRICE'])
plt.show()


# In[775]:


# we can see that the sales price is highly skewed, decided to remove price which is larger than 2 * 10^9
df = df[df['SALE PRICE']<2*10**9]
df.shape


# In[776]:


# Plot the distribution of SALE PRICE
plt.boxplot(df['SALE PRICE'])
plt.show()


# In[777]:


percent_75 = np.percentile(df['SALE PRICE'], 75)
np.percentile(df['SALE PRICE'], 1)
df = df[df['SALE PRICE'] < percent_75]
plt.hist(df['SALE PRICE'], bins= 100)
plt.show()


# In[778]:


# Remove the records with too low sales price
df = df[df['SALE PRICE']>100000]
plt.hist(df['SALE PRICE'], bins= 50)
plt.show()


# ### Zip Code & Square Feet

# In[779]:


# remove data records with zip code not in new york(less than 10000)
df= df[df['ZIP CODE']>10000]
df= df[df['GROSS SQUARE FEET']>0]
df= df[df['LAND SQUARE FEET']>0]


# In[780]:


df['ZIP CODE'].describe()


# In[781]:


df.shape


# In[785]:


df['ZIP CODE']=df['ZIP CODE'].astype('object')
df['BOROUGH']=df['BOROUGH'].astype('object')


# ## Data Overview

# In[786]:


df.shape


# In[787]:


df.info()


# In[788]:


df.describe()


# # **Exploratory Data Analysis**

# ## Describe Numeric Data

# In[789]:


## Explore numerical features
df.describe(include=[np.number])


# In[790]:


col=['BOROUGH','ZIP CODE','BLOCK','LOT','ZIP CODE','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','TAX CLASS AT TIME OF SALE','SALE PRICE']
df[col]


# ## Describe Categorical Data

# In[791]:


df_categorical.describe()


# In[792]:


df['BOROUGH'].unique()


# In[793]:


df['BOROUGH'].hist()


# In[794]:


df['TAX CLASS AT PRESENT'].unique()


# In[795]:


df['TAX CLASS AT PRESENT'].hist()


# In[796]:


#  TAX CLASS AT PRESENT
pivot1 = df.groupby("TAX CLASS AT PRESENT")['SALE PRICE'].median()
pivot1


# In[797]:


pivot1.plot(kind='bar', color='b')


# In[798]:


#  TAX CLASS AT TIME OF SALE
df['TAX CLASS AT TIME OF SALE'].unique()


# In[799]:


pivot2 = df.groupby('TAX CLASS AT TIME OF SALE')['SALE PRICE'].median()
pivot2


# In[800]:


pivot2.plot(kind='bar', color='r')


# In[801]:


#BOROUGH
df['BOROUGH'].unique()


# In[802]:


pivot3_1 = df.groupby('BOROUGH')['SALE PRICE'].median()
pivot3_1


# In[803]:


pivot3_1.plot(kind='bar', color='g')


# In[804]:


pivot3_2=df.groupby('ZIP CODE')['SALE PRICE'].median()
pivot3_2


# In[805]:


pivot3_2.plot(kind='area',color='green')


# In[806]:


#BUILDING CLASS CATEGORY
print(df['BUILDING CLASS CATEGORY'].nunique())


# In[807]:


pivot4 = df.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].median()
pivot4


# In[808]:


plt.figure(figsize=(6,6))
pivot4.plot(kind='bar', color='Green')


# # **Optimized Model Preparation**

# ## One-Hot Encoding

# In[809]:


#Features to be one-hot encoded
one_hot_features = ['BOROUGH', 'ZIP CODE','BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']
#Converting categorical variables into indicator variables
one_hot_encoded = pd.get_dummies(df[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)


# In[810]:


# Replacing categorical columns with dummies
fdf = df.drop(one_hot_features,axis=1)
fdf = pd.concat([fdf, one_hot_encoded] ,axis=1)
fdf.info()


# ## Correlation Matrix

# In[811]:


correlation_matrix_numerical=df[col].corr(method='pearson')
correlation_matrix_numerical


# In[665]:


correlation_matrix_categorical=fdf.corr(method='pearson')
correlation_matrix_categorical


# ## Numeric Features Heatmap

# In[812]:


plt.subplots(figsize = (12,10))
sns.heatmap(correlation_matrix_numerical)


# # **Optimized Models**

# ## Data Preview

# In[814]:


df=fdf


# In[815]:


df.shape


# In[816]:


df.info()


# ## Create Dataset

# In[817]:


df_temp = df.sample(n=int(df.shape[0]*0.1), random_state=1)
# Create dataset
X, y = df_temp.drop(['SALE PRICE', 'SALE DATE'], axis=1), df_temp['SALE PRICE']


# ## Random Forest

# In[818]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate


# In[819]:


#define MAPE function to calculate mean absolute percentage error
import numpy as np
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)


# In[820]:


rf_regressor = RandomForestRegressor(min_samples_leaf=10)
scores_rf = cross_validate(rf_regressor, X, y, cv=10,
                              scoring=make_scorer(MAPE),
                              return_train_score=True)


# In[821]:


print('The mape result for random forest is {}'.format(np.min(scores_rf['test_score'])))


# ## Adaboost

# In[675]:


from sklearn.ensemble import AdaBoostRegressor


# In[676]:


regr = AdaBoostRegressor(random_state=0, n_estimators=500)
scores_adaboost = cross_validate(regr, X, y, cv=5,
                              scoring=make_scorer(MAPE),
                              return_train_score=True)


# In[677]:


print('The best score for Adaboost is {}'.format(np.min(scores_adaboost['test_score'])))


# ## Lightgbm

# In[678]:


import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = lgb.LGBMRegressor(num_leaves=1000,
                        learning_rate=0.01,
                        n_estimators=300)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
       early_stopping_rounds=100)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The mape of prediction is:', MAPE(y_test, y_pred))


# ## XGBoost

# ### Build MAE & MAPE Functions

# In[679]:


import numpy as np
def xgboost_MAE(pred,  dtrain: xgb.DMatrix):
    true = dtrain.get_label()
    diff = np.abs(np.array(true) - np.array(pred))
    return 'mae', float(np.mean(diff))

def xgboost__MAPE(pred,  dtrain: xgb.DMatrix):
    true = dtrain.get_label()
    diff = np.abs(np.array(true) - np.array(pred))
    return 'mape', float(np.exp(np.mean(diff))-1)
 


# ### Calculate MAE of Original Price

# In[849]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error
clf = xgb.XGBRegressor(learning_rate=0.04,
                  n_estimators=600,
                  max_depth=5, colsample_bylevel=0.9)


# In[850]:


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)
clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=xgboost_MAE,
        verbose=True )

print('Starting predicting...')
# predict
y_pred = clf.predict(X_test)
# eval
mae = mean_absolute_error(y_test, y_pred)
print('The MAE of predictions on original sales price are:',mae)


# ### Calculate MAE on Log Price ( MAPE on Original Price)

# In[851]:


X_train, X_test, ylg_train, ylg_test = train_test_split(X, np.log(y), test_size=0.2, random_state=42)
clf.fit(X_train, ylg_train,
        eval_set=[(X_train, ylg_train), (X_test, ylg_test)],
        eval_metric=xgboost__MAPE,
        verbose=True )

print('Starting predicting...')
# predict
ylg_pred = clf.predict(X_test)
# eval
mae_logpx =mean_absolute_error(ylg_test, ylg_pred)
print('The mae of log prices are:',mae_logpx)


# # **Conclusion**

# We tried four models on the real estate sales price dataset. After being cleaned, the dataset contains more than 40k pieces of data with around 300 features. Measures taken in order to gain better performance include adjusting paramters, passing in customized evaluation functions, computing log on original price etc. 
# 
# The best performance acheived by our models is around MAE of 70k dollars, MAPE of 0.25. 
