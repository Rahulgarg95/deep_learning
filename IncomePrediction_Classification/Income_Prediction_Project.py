#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,ElasticNetCV,LassoCV,RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


import warnings

warnings.filterwarnings(action='ignore')


# In[3]:


#Loading data in pandas
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])


# ### Basic dataframe details and stats
# 
# ----

# In[4]:


df.head()


# In[5]:


#Getting an idea of total records and feature in dataframe
df.shape


# In[6]:


#Checking datatypes and no of null values in df
df.info()


# In[7]:


#Getting basic stats for numerical feature
df.describe().T


# In[8]:


#Checking if any null values are present per column/feature
df.isna().sum()


# ***NULL values are not present in this data, hence no need for imputation techniques.***

# ### Exploratory Data Analysis

# 1. ***Exploring categorical features and checking how can same be converted to numrical features***

# In[9]:


#Custom Function for one hot encoding and after operations
def custom_encoding(df,col):
    dummies=pd.get_dummies(df[col],drop_first=True)
    df=df.drop(col,axis=1)
    df=pd.concat([df,dummies],axis=1)
    return df


# In[10]:


#Feature: workclass


# In[11]:


df['workclass'].value_counts()


# In[12]:


df.workclass=df.workclass.str.strip()


# **Found ? in many rows hence replacing the same with NaN for now.**

# In[13]:


for x in ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','salary']:
    df[x]=df[x].str.strip()
    df[x]=df[x].replace(' ','')


# In[14]:


df=df.replace('?',np.nan).dropna()


# In[15]:


df.shape


# In[16]:


df['workclass'].isna().sum()


# In[17]:


plt.figure(figsize=(12,6))
sns.countplot('workclass',data=df,hue='salary')


# In[18]:


df=custom_encoding(df,'workclass')


# In[19]:


df.columns


# In[20]:


# Dependent Feature: salary


# In[21]:


df['salary'].value_counts()


# In[22]:


sns.countplot('salary',data=df)


# **Seems like the dataset is imbalanced.**

# In[23]:


#df.to_csv('tmp.csv',index=False)


# In[24]:


df['salary']=df['salary'].map({'<=50K':1,'>50K':0})


# In[25]:


# Feature: education & education-num


# In[26]:


df['education'].value_counts()


# In[27]:


df['education-num'].value_counts()


# **Features education and education-num seems to be same dropping education column**

# In[28]:


plt.figure(figsize=(16,6))
sns.countplot('education',data=df,hue='salary')


# In[29]:


df=df.drop(columns=['education'])


# In[30]:


#Feature: marital-status


# In[31]:


df['marital-status'].value_counts()


# In[32]:


plt.figure(figsize=(16,6))
sns.countplot('marital-status',data=df,hue='salary')


# In[33]:


df=custom_encoding(df,'marital-status')


# In[34]:


# Feature:occupation


# In[35]:


df['occupation'].value_counts()


# In[36]:


plt.figure(figsize=(16,6))
sns.countplot('occupation',data=df,hue='salary')
plt.xticks(rotation=45)
plt.show()


# In[37]:


df=custom_encoding(df,'occupation')


# In[38]:


# Feature: relationship


# In[39]:


df['relationship'].value_counts()


# In[40]:


plt.figure(figsize=(16,6))
sns.countplot('relationship',data=df,hue='salary')
plt.xticks(rotation=45)
plt.show()


# In[41]:


df=custom_encoding(df,'relationship')


# In[42]:


# Feature:race


# In[43]:


df['race'].value_counts()


# In[44]:


plt.figure(figsize=(10,4))
sns.countplot('race',data=df,hue='salary')
plt.xticks(rotation=45)
plt.show()


# In[45]:


df=custom_encoding(df,'race')


# In[46]:


#Feature: sex


# In[47]:


plt.figure(figsize=(10,4))
sns.countplot('sex',data=df,hue='salary')
plt.xticks(rotation=45)
plt.show()


# In[48]:


df['sex']=df['sex'].map({'Male':0,'Female':1})


# In[49]:


#Feature: native-country


# In[50]:


df['native-country'].value_counts()


# In[51]:


df['native-country']=df['native-country'].apply(lambda x:'US' if('United-States' in x) else 'Non-US')


# In[52]:


df['native-country'].value_counts()


# In[53]:


plt.figure(figsize=(7,5))
sns.countplot('native-country',data=df,hue='salary')
plt.xticks(rotation=45)
plt.show()


# In[54]:


df['native-country']=df['native-country'].map({'US':0,'Non-US':1})


# In[55]:


df.columns


# In[56]:


df.dtypes


# ### Column Standardization

# In[57]:


X=df.drop('salary',axis=1)
y=df['salary']


# In[58]:


sc=StandardScaler()


# In[59]:


tmp_scaled=sc.fit_transform(X[['age','fnlwgt','capital-gain','capital-loss','hours-per-week']])


# In[60]:


X_tmp=X.copy()


# In[61]:


X_tmp[['age','fnlwgt','capital-gain','capital-loss','hours-per-week']]=tmp_scaled


# In[62]:


X_scaled=X_tmp.copy()


# In[63]:


X_scaled.to_csv('scaled_tmp.csv',index=False)


# In[64]:


from sklearn.decomposition import PCA
pca=PCA()
pca.fit(X_scaled)
#Plotting to get an idea regarding the count of components required to expained the variance
plt.grid()
plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# ### Splitting data using train_test_split

# In[65]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.25,random_state=365)


# In[66]:


x_train.shape


# In[67]:


x_test.shape


# ### Logistic Regression
# 
# ----

# In[68]:


regression=LogisticRegression()


# In[69]:


regression.fit(x_train,y_train)


# In[70]:


print('Coeff Values: ',regression.coef_)
print('Intercept Value: ',regression.intercept_)


# In[71]:


regression.score(x_train,y_train)


# In[72]:


regression.score(x_test,y_test)


# In[73]:


y_pred=regression.predict(x_test)


# In[74]:


y_test


# In[75]:


y_pred


# **Model does not seems to be overfitting**

# In[76]:


from sklearn.metrics import classification_report,log_loss,confusion_matrix


# In[77]:


print(classification_report(y_test,y_pred))


# In[78]:


print(log_loss(y_test,y_pred))


# In[79]:


print(confusion_matrix(y_test,y_pred))


# In[80]:


from sklearn.linear_model import SGDClassifier
sgdmodel=SGDClassifier(random_state=365)


# In[81]:


regression.fit(x_train,y_train)


# In[82]:


print(regression.score(x_train,y_train))
print(regression.score(x_test,y_test))


# In[83]:


#Hypertuning SGD Classifier


# In[84]:


alpha = [10 ** x for x in range(-5, 2)] # hyperparam for SGD classifier.
for x in alpha:
    sgdmodel=SGDClassifier(alpha=x,random_state=365)
    sgdmodel.fit(x_train,y_train)
    print(x,': ',sgdmodel.score(x_train,y_train))
    print(x,': ',sgdmodel.score(x_test,y_test))


# ### Decision Tree
# 
# ----

# In[85]:


clf=DecisionTreeClassifier()


# In[86]:


clf.fit(x_train,y_train)


# In[87]:


print('Train Data Score: ',clf.score(x_train,y_train))
print('Test Data Score: ',clf.score(x_test,y_test))


# In[88]:


#Model is getting overfitted


# In[89]:


grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,20,2),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random']   
}


# In[90]:


from sklearn.model_selection import GridSearchCV


# In[91]:


grid_search = GridSearchCV(clf,param_grid=grid_param,cv=5,verbose=5,n_jobs=-1)
grid_search.fit(x_train,y_train)


# In[92]:


best_dict=grid_search.best_params_
print(best_dict)


# In[93]:


clf1=DecisionTreeClassifier(criterion=best_dict['criterion'],max_depth=best_dict['max_depth'],min_samples_leaf=best_dict['min_samples_leaf'],min_samples_split=best_dict['min_samples_split'],splitter=best_dict['splitter'])


# In[94]:


clf1.fit(x_train,y_train)


# In[95]:


print(clf1.score(x_train,y_train))
print(clf1.score(x_test,y_test))


# ### RandomForest Classifier
# 
# ----

# In[96]:


from sklearn.ensemble import RandomForestClassifier


# In[97]:


rf_clf=RandomForestClassifier()


# In[98]:


rf_clf.fit(x_train,y_train)


# In[99]:


print(rf_clf.score(x_train,y_train))
print(rf_clf.score(x_test,y_test))


# In[100]:


grid_params = {"n_estimators" : [30,70,110,150,190,230],
              "max_depth" : range(1,10,2),
              "min_samples_leaf" : range(1,10,1),
              "min_samples_split" : range(2,10,1)
              }


# In[101]:


grid_search = GridSearchCV(rf_clf,param_grid=grid_params,cv=5,verbose=5,n_jobs=-1)
grid_search.fit(x_train,y_train)


# In[102]:


grid_search.best_params_


# In[103]:


rf_clf1=RandomForestClassifier(max_depth=9,max_features='auto',min_samples_leaf=3,min_samples_split=6,n_estimators=70)


# In[104]:


rf_clf1.fit(x_train,y_train)


# In[105]:


print(rf_clf1.score(x_train,y_train))
print(rf_clf1.score(x_test,y_test))


# In[107]:


import xgboost as xgb


# In[110]:


xgb_model=xgb.XGBClassifier()


# In[111]:


xgb_model.fit(x_train,y_train)


# In[112]:


xgb_model.score(x_train,y_train)


# In[113]:


xgb_model.score(x_test,y_test)


# In[115]:


grid_params={
        'n_estimators': [70,90,110,130,150,190],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
}


# In[116]:


grid_search = GridSearchCV(xgb_model,param_grid=grid_params,cv=5,verbose=5,n_jobs=-1)
grid_search.fit(x_train,y_train)


# In[117]:


grid_search.best_params_


# In[119]:


xgb_model1=xgb.XGBClassifier(colsample_bytree=1.0,gamma=5,max_depth=4,min_child_weight=1,n_estimators=90,subsample=0.8)


# In[120]:


xgb_model1.fit(x_train,y_train)


# In[121]:


xgb_model1.score(x_train,y_train)


# In[122]:


xgb_model1.score(x_test,y_test)


# In[124]:


xgb_model1.feature_importances_


# In[132]:


#Features with importance are plotted
plt.figure(figsize=(12,10))
plt.barh(X_scaled.columns, xgb_model1.feature_importances_)


# In[129]:


y_pred=xgb_model1.predict(x_test)


# In[130]:


print(classification_report(y_test,y_pred))


# In[131]:


print(confusion_matrix(y_test,y_pred))


# In[133]:


from tabulate import tabulate


# In[134]:


df_model=pd.DataFrame({'Model_Name':['LogisticRegression','SGDClassfier(Hinge Loss)','DecisionTree','RandomForestClassifier','XGBoost'],
                      'Test Performance':[0.8510807585200901,0.8494894576316139,0.8566503116297574,0.8574459620739955,0.8669937674048535]})


# In[136]:


print(tabulate(df_model, headers='keys', tablefmt='psql'))

