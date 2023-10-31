#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\omkar\Downloads\College ML\Salary_Data.csv")
df.head()


# In[3]:


plt.scatter(df["YearsExperience"],df["Salary"])
plt.xlabel("years of experience")
plt.ylabel("salary")


# In[4]:


x = df.iloc[:,0:1]
y = df.iloc[:,-1]


# In[5]:


x.head()


# In[6]:


y.head()


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 2)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


lr = LinearRegression() #call object


# In[10]:


lr.fit(x_train,y_train)       #trains our data


# In[11]:


x_test


# In[12]:


y_test


# In[13]:


#test model


# In[14]:


lr.predict(x_test.iloc[0])


# In[15]:


#above we get error because " Expected 2D array, got 1D array instead: "


# In[22]:


lr.predict(x_test.iloc[5].values.reshape(1,1))


# In[23]:


# regression line


# In[26]:


plt.scatter(df["YearsExperience"],df["Salary"])
plt.plot(x_train,lr.predict(x_train),color = 'red')
plt.xlabel("years of experience")
plt.ylabel("salary")


# In[27]:


#this is found by lr , and that is indicating me that it is the best fit line


# In[28]:


m = lr.coef_


# In[29]:


b = lr.intercept_


# In[30]:


m


# In[31]:


b


# In[ ]:




