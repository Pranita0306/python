#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
Boston=pd.read_csv('./BostonHousing.csv')
Boston.head()


# In[17]:


import pandas as pd


# In[34]:


# Applying LR algorithm
y=Boston[['medv']] # dependant variable
x=Boston[['crim']]  # independant variable
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_test.head()


# In[35]:


y_pred[0:8]


# In[36]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


# In[37]:


x=Boston[['lstat']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr2=LinearRegression()
lr2.fit(x_train,y_train)
y_pred2=lr2.predict(x_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred2)


# In[ ]:





# In[ ]:




