#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# In[2]:


house_price_dataset = sklearn.datasets.load_boston()


# In[3]:


print(house_price_dataset)


# In[4]:


house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)


# In[5]:


house_price_dataframe.head()


# In[6]:


X = house_price_dataframe['Price'] = house_price_dataset.target
house_price_dataframe.head()


# In[7]:


house_price_dataframe.shape


# In[8]:


house_price_dataframe.isnull().sum()


# In[9]:


house_price_dataframe.describe()


# In[10]:


correlation = house_price_dataframe.corr()


# In[11]:


plt.figure(figsize = (10, 10))
sns.heatmap(correlation, cbar = False, square = True, fmt = '.1f', annot = True, annot_kws = {'size':8}, cmap = 'Blues')


# In[12]:


X = house_price_dataframe.drop(['Price'], axis = 1)
Y = house_price_dataframe['Price']

print(X)
print(Y)


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# In[15]:


#loading the model
linear_model = LinearRegression()


# In[16]:


#train the model
linear_model.fit(X_train, Y_train)


# In[17]:


# accuracy on prediction on training data
training_data_prediction = linear_model.predict(X_train)


# In[18]:


print(training_data_prediction)


# In[19]:


#R square error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

#Mean absolute error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)


# In[20]:


print("R square error :", score_1)
print("Mean Absolute Error:", score_2)


# In[21]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel('actual prices')
plt.ylabel('predicted prices')
plt.title('actual prices vs prediction prices')
plt.show()


# In[22]:


# accuracy on prediction on test data
test_data_prediction = linear_model.predict(X_test)


# In[23]:


#R square error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

#Mean absolute error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R square error :", score_1)
print("Mean Absolute Error:", score_2)


# In[ ]:





# In[ ]:




