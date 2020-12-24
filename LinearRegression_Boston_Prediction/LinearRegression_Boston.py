#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
BOSTON DATASET - Linear Regression

@Author - Rahul Garg (rahu.garg3@hpe.com)
'''


# In[66]:


#Importing Required Modules
import os,sys
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import sklearn
from sklearn.datasets import load_boston
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


boston=load_boston()
df_bos=pd.DataFrame(boston.data,columns=list(boston.feature_names))


# In[68]:


df_bos.head()


# In[69]:


df_bos['Price']=list(boston.target)


# In[70]:


df_bos.head()


# In[71]:


#Boston Dataset Description
print(boston.DESCR)


# In[72]:


print(df_bos.shape)


# In[73]:


print(df_bos.info())


# **No NULL Values are present in boston dataset.**

# In[74]:


df_bos.describe()


# In[75]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df_bos:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(df_bos[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# **Skewness is observed in many features let us visualize if any outliers are present**

# In[76]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df_bos, width= 0.5,ax=ax,  fliersize=3)
'''plt.figure(figsize=(15,10), facecolor='white')
df_bos.boxplot()'''


# **Outliers are observed in the dataset as can be seen in boxplot**

# In[77]:


df_bos.columns


# In[78]:


#Handling Outliers
q = df_bos['CRIM'].quantile(0.98)
data_cleaned = df_bos[df_bos['CRIM']<q]

q = data_cleaned['ZN'].quantile(0.98)
data_cleaned  = data_cleaned[data_cleaned['ZN']<q]

q = data_cleaned['INDUS'].quantile(0.99)
data_cleaned  = data_cleaned[data_cleaned['INDUS']<q]

q = data_cleaned['CHAS'].quantile(0.99)
data_cleaned  = data_cleaned[data_cleaned['CHAS']<q]

q = data_cleaned['B'].quantile(0.95)
data_cleaned  = data_cleaned[data_cleaned['B']<q]

q = data_cleaned['LSTAT'].quantile(0.98)
data_cleaned  = data_cleaned[data_cleaned['LSTAT']<q]

q = data_cleaned['PTRATIO'].quantile(0.99)
data_cleaned  = data_cleaned[data_cleaned['PTRATIO']<q]

q = data_cleaned['DIS'].quantile(0.99)
data_cleaned  = data_cleaned[data_cleaned['DIS']<q]

q = data_cleaned['AGE'].quantile(0.99)
data_cleaned  = data_cleaned[data_cleaned['AGE']<q]


# In[79]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data_cleaned:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data_cleaned[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[80]:


data_cleaned.shape


# In[85]:


y = df_bos['Price']
X =df_bos.drop(columns = ['Price'])

'''y=data_cleaned['Price']
X=data_cleaned.drop(columns=['Price'])'''


# In[86]:


plt.figure(figsize=(20,30), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=15 :
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(X[column],y)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Price',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[88]:


from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# In[89]:


#Scaling the features
scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[90]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_scaled
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = X.columns


# In[92]:


vif.sort_values(by='VIF')


# **As 5 is the VIF threshold value for features hence we see RAD and TAX columns have high multicollinearity.**

# In[95]:


# Lets look at the correlation matrix of our data.
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111)
sns.heatmap(X.corr(),annot=True)


# Our target variable, seems to be highly correlated, with LSTAT and RM, which makes sense, as these two are very important factors for house pricing, but there seems to be a lot of multicollinearity as well.
# 
# The issue here is, that there is a lot of collinearity between our predictor variables, for example DIS is highly correlated to INUDS, INOX and AGE.
# 
# This is not good, as multicollinearity can make our model unstable, we need to look at it a little more, before modeling our data, I have explained, the probem of multicollinearity below.

# In[96]:


#Applying PCA to remove the collinearily between the data


# In[97]:


from sklearn.decomposition import PCA


# In[99]:


pca=PCA(n_components=13)


# In[100]:


X_pca=pca.fit_transform(X)


# In[101]:


X_pca


# ###### Checking Multicollinearity after PCA using VIF and HeatMap

# In[103]:


variables = X_pca
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = X.columns


# In[104]:


vif.sort_values(by='VIF')


# In[106]:


df_pca=pd.DataFrame(X_pca,columns=list(X.columns))


# In[112]:


fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111)
sns.heatmap(df_pca.corr(),annot=True)


# In[113]:


df_pca


# ##### Linear Regression Application

# In[114]:


x_train,x_test,y_train,y_test=train_test_split(X_pca,y,test_size=0.25,random_state=355)


# In[116]:


x_train.shape


# In[117]:


x_test.shape


# In[118]:


regression=LinearRegression()
regression.fit(x_train,y_train)


# In[121]:


print('Coeff Values: ',regression.coef_)
print('Intercept Value: ',regression.intercept_)


# In[128]:


from sklearn.metrics import mean_squared_error,r2_score


# In[126]:


#Training model on training data and predicting on training data
y_pred=regression.predict(x_train)
print(y_train.to_list())
print(list(y_pred))


# In[131]:


r2 = r2_score(y_train,y_pred)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
print('R2 Score: ', r2)
print('Root Mean Sq Score: ',rmse)


# In[134]:


#Training model on training data and predicting on test data
y_pred=regression.predict(x_test)
print(y_train.to_list())
print(list(y_pred))
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('R2 Score: ', r2)
print('Root Mean Sq Score: ',rmse)


# In[136]:


st_file='stand_scaler.pickle'
pickle.dump(scaler, open(st_file, 'wb'))


# In[137]:


# saving the model to the local file system
filename = 'linear_model.pickle'
pickle.dump(regression, open(filename, 'wb'))


# In[138]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV


# In[144]:


# Lasso Regularization
# LassoCV will return best alpha and coefficients after performing 10 cross validations
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
lasscv.fit(x_train, y_train)


# In[145]:


# best alpha parameter
alpha = lasscv.alpha_
print(alpha)

#now that we have best parameter, let's use Lasso regression and see how well our data has fitted before

lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train, y_train)

print(lasso_reg.score(x_test, y_test))


# **After L1 regularization RMS value is same as that of normal Linear Regression. This means our model does not overfit.**

# In[146]:


# Using Ridge regression model
# RidgeCV will return best alpha and coefficients after performing 10 cross validations. 
# We will pass an array of random numbers for ridgeCV to select best alpha from them

alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(x_train, y_train)

ridge_model = Ridge(alpha=ridgecv.alpha_)
ridge_model.fit(x_train, y_train)

ridge_model.score(x_test, y_test)


# **After L2 regularization also RMS value is same as that of normal Linear Regression. This means our model does not overfit.**

# In[148]:


# Elastic net
elasticCV = ElasticNetCV(alphas = None, cv =10)
elasticCV.fit(x_train, y_train)

print(elasticCV.alpha_)
# l1_ration gives how close the model is to L1 regularization, below value indicates we are giving equal
#preference to L1 and L2
print(elasticCV.l1_ratio)

elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)
elasticnet_reg.fit(x_train, y_train)
elasticnet_reg.score(x_test, y_test)


# **So, we can see by using different type of regularization, we still are getting the same r2 score. That means our OLS model has been well trained over the training data and there is no overfitting.**
