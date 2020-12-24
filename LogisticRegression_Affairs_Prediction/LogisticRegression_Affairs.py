#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


dta=sm.datasets.fair.load_pandas().data


# In[3]:


dta


# In[4]:


dta['affair'] = (dta.affairs >0).astype(int)


# In[5]:


#Checking the distribution of dependent feature
dta.affair.value_counts()


# **Seems to be an imbalanced dataset as one category has double the records of other**

# In[6]:


#Checking for NULL Values
dta.info()


# **NULL values are not present in above dataset**

# In[7]:


#Getting a summary of data
dta.describe()


# In[8]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in dta:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(dta[column])
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[9]:


# let's see how data is distributed for every column and each category.
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in dta.drop(columns=['affair']):
    if plotnumber<=16:
        ax = plt.subplot(4,4,plotnumber)
        sns.violinplot(y=column,x='affair',data=dta)
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# **Some of the attributes(rate_marriage,age,yrs_married,children,education) give us an idea how we can distribute the data**

# In[10]:


y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',dta, return_type="dataframe")


# In[11]:


X = X.rename(columns =
{'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})


# In[12]:


#Flattening the array
y=np.ravel(y)


# In[13]:


y


# In[14]:


X.head()


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)


# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif


# In[18]:


from sklearn.decomposition import PCA


# In[19]:


pca=PCA(n_components=16)
pca.fit(X_scaled)
X_scaled=pca.transform(X_scaled)


# In[20]:


'''vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif'''


# In[21]:


from sklearn.model_selection import GridSearchCV


# In[22]:


tuned_parameters = [{'C': [10**-4, 10**-2, 10**0, 10**2, 10**4]}]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.80, random_state=355)


# In[23]:


#Creating a logistic regression model
model=LogisticRegression()
model


# In[24]:


model.fit(X_train,y_train)


# In[25]:


print(model.intercept_)
print(model.coef_)


# In[26]:


pd.DataFrame(list(zip(X.columns,np.transpose(model.coef_))))


# In[27]:


y_pred=model.predict(X_test)


# In[28]:


print(model)


# In[29]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[30]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[31]:


# Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[32]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[33]:


# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[34]:


# Precison
Precision = true_positive/(true_positive+false_positive)
Precision


# In[35]:


# Recall
Recall = true_positive/(true_positive+false_negative)
Recall


# In[36]:


# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[37]:


# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
auc


# In[38]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)


# In[39]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[40]:


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
scores, scores.mean()


# In[ ]:




