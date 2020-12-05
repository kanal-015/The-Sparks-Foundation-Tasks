#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation (GRIP, DEC-2020)
# ## Data Science and Business Analytics Tasks
# ## Name: Soni Kanal Bhadreshkumar
# ## Task-1: Prediction using Supervised Learning

# In[76]:


#Step-1: Let's import libraries

import pandas as pd
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


# In[77]:


#Step-2: Let's read the data

data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print(data)


# In[78]:


#Step-3: Let's check if there exists any missing data 

data.isnull().sum()


# As we saw above, that no missing data is found. Thus, the process of Data Cleaning is not needed in this case.

# In[79]:


#Step-4: Plotting the distribution of scores that means the input data
data.plot(x='Hours', y='Scores', style='ro')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours that student Studied')  
plt.ylabel('Total Percentage Score')  
plt.show()


# In[80]:


#Step-5: Let's divide the data into inputs (attributes) and outputs (labels)

x=data.iloc[:, :-1].values #input
y=data.iloc[:,1].values #output


# In[81]:


#Step-6: Splitting the data into train and test set for prediction

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7)


# In[82]:


#Step-7: Let's fit simple Linear Regression

lr=LinearRegression()
lr.fit(x_train,y_train)


# In[83]:


#Step-8: Let's plot the regression line for training set

line = lr.coef_*x+lr.intercept_

# Plotting for the test data
plt.figure(figsize=(10,6))
plt.scatter(x, y, color='brown')
plt.plot(x, line, color='green')
plt.title('Score vs Hours of the student who is studying')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[84]:


#Step-9: Let's plot hte regression line for test data

plt.figure(figsize=(10,6))
plt.scatter(x_test,y_test,color='grey')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('Score vs Hours of the student who is studying')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[85]:


#Step-10: Let's predict the values 

y_pred=lr.predict(x_test)
print(y_pred)
data=pd.DataFrame({'Actual value of Y': y_test, 'Predicted value of Y': y_pred})
print(data)
 


# In[86]:


#Step-11: Let's plot the bar graph for actual and predicted values of y

data.plot()


# In[87]:


#Step-12: Let's check the accuracy of the model

model=LinearRegression()
model.fit(x_train,y_train)
print('r2: ',r2_score(y_test, y_pred))


# Thus, we got 95.05% accuracy for our model which means that we have done perfect test on our data. This algorithm proved good for the given data.
