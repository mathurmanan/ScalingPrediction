#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format 
import matplotlib.pyplot as plt
sns.set(style = 'white')
df = pd.read_csv('scale.csv')
df.head()


# In[2]:


df.dtypes


# In[4]:


df.Innovation Intensity = pd.numeric(df.InnovationIntensity, errors='coerce')


# In[7]:


df.InnovationIntensity = pd.to_numeric(df.InnovationIntensity, errors='coerce')


# In[8]:


df.dtypes


# In[9]:


df.RevenueGrowth = pd.to_numeric(df.RevenueGrowth, errors='coerce')


# In[10]:


df.dtypes


# In[11]:


df.isnull().sum()


# In[12]:


df.dropna(inplace = True)


# In[13]:


df.isnull().sum()


# In[14]:


df2 = df.iloc[:,1:]


# In[16]:


df2.isnull().sum()


# In[17]:


df2.dtypes


# In[20]:


df2.No_VC = pd.CategoricalDtype(df2.No_VC, errors='coerce')


# In[21]:


df2.No_VC = df2.No_VC.astype('str')


# In[22]:


df2.dtypes


# In[23]:


#Converting all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
df_dummies.head()


# In[24]:


df2.dtypes


# In[25]:


df2.head()


# In[26]:


#Get Correlation of Churn with other variables
plt.figure(figsize=(15,8))
df_dummies.corr()['Scale'].sort_values(ascending = False).plot(kind='bar')


# In[27]:


ax = (df2['No_VC'].value_counts()*100.0 /len(df2)).plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )                                                                           
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('No of VC',fontsize = 12)
ax.set_title('% of VC', fontsize = 12)


# In[28]:


df2 = df.iloc[:,2:]


# In[29]:


df2.head()


# In[30]:


plt.figure(figsize=(15,8))
df_dummies.corr()['Scale'].sort_values(ascending = False).plot(kind='bar')


# In[31]:


df2.dtypes


# In[32]:


df2 = df.iloc[:,1:]


# In[33]:


df2.dtypes


# In[34]:


plt.figure(figsize=(15,8))
df_dummies.corr()['Scale'].sort_values(ascending = False).plot(kind='bar')


# In[36]:


df2.drop(['RevenueGrowth'], axis=1)


# In[39]:


plt.figure(figsize=(15,8))
df_dummies.corr()['Scale'].sort_values(ascending = False).plot(kind='bar')


# In[41]:


ax = df2['No_VC'].value_counts().plot(kind = 'bar',rot = 0, width = 0.3)
ax.set_ylabel('# of firms')
ax.set_title('# of firms by VC participation')


# In[43]:


#Analysing the predictor varibale

colors = ['#4D3425','#E4512B']
ax = (df2['Scale'].value_counts()*100.0 /len(df2)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors,
                                                                         figsize = (8,6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% of firms',size = 14)
ax.set_xlabel('Scale',size = 14)
ax.set_title('Scale Percentage', size = 14)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-4.0,             str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold',
           size = 14)


# In[44]:


#Scale vs Marketing Intensity

sns.boxplot(x = df2.Scale, y = df2.MarketingIntensity)


# In[45]:


#Scale vs Innovation Intensity

sns.boxplot(x = df2.Scale, y = df2.InnovationIntensity)


# In[46]:


#Scale vs No. of VC

sns.boxplot(x = df2.Scale, y = df2.No_VC)


# In[57]:


#Scale vs Age of Board 

sns.boxplot(x = df2.Scale, y = df2.Board_Age)


# In[68]:


#Logistic Regression

# We will use the data frame where we had created dummy variables
y = df_dummies['Scale'].values
X = df_dummies.drop(columns = ['Scale', 'RevenueGrowth'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In[69]:


# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[70]:


# Running logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[71]:


from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))


# In[72]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# In[73]:


print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))


# In[77]:


from joblib import dump, load
#model = joblib.dump('Scale-Predictor-Logistic.joblib')
#dump(clf, 'filename.joblib')

dump(model, 'Scale-Predictor-Logistic.joblib')


# In[78]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[79]:


dump(model_rf, 'Scale-Predictor-Random_Forest.joblib')


# In[80]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# In[81]:


# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)
model.fit(X_train,y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[ ]:




