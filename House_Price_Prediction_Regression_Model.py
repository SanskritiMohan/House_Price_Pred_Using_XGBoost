#!/usr/bin/env python
# coding: utf-8

# In[1]:


#House price prediction
#Regression model
#Boston House Price Dataset- UCI ML Repo

#Work Flow
#House price data-Data pre processing- Data Analysis-Train test split
#XGBoost Regressor-Evaluation


# In[ ]:


#Import the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[5]:


#importing the Boston House Price Dataset
#house_price_dataset = sklearn.datasets.load_boston()
#removed due to ethical reasons
#use california dataset


# In[6]:


# Importing the necessary module from scikit-learn
from sklearn.datasets import fetch_california_housing

# Loading the California House Price Dataset
house_price_dataset = fetch_california_housing()

# Displaying the dataset description
print(house_price_dataset.DESCR)


# In[7]:


print(house_price_dataset)


# In[12]:


#loading the dataset to a panda dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data,columns = house_price_dataset.feature_names)


# In[13]:


#Print first 5 rows of our DataFrame
house_price_dataframe.head()


# In[14]:


#add the target column (price) to the dataframe
house_price_dataframe['price'] = house_price_dataset.target


# In[16]:


house_price_dataframe.head()


# In[18]:


#checking number of rows and columns in the dataframe
house_price_dataframe.shape


# In[20]:


#check for missing values
house_price_dataframe.isnull().sum()


# In[21]:


#statistical measures of the dataset
house_price_dataframe.describe()


# In[22]:


#Understanding the correlation between various features in the dataset
#1. Positive correlation- one increases other increases
#2. Negative correlation- one inc other dec


# In[24]:


correlation = house_price_dataframe.corr()


# In[32]:


#constructing a heatmap to understand the correlation
plt.figure(figsize = (10,10))
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size':8}, cmap = 'Blues')


# In[33]:


#splitting the data and target
X = house_price_dataframe.drop(['price'], axis = 1) #1 for col 0 for row
Y = house_price_dataframe['price']


# In[34]:


print(X)
print(Y)


# In[36]:


#splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)


# In[41]:


print(X.shape, X_train.shape, X_test.shape)


# In[45]:


#model training
#XGBoost Regressor- decision tree based ensemble learner


# In[52]:


get_ipython().system('pip install xgboost')
from xgboost import XGBRegressor


# In[53]:


#loading the model
model = XGBRegressor()


# In[54]:


#training the model with x_train
model.fit(X_train, Y_train)


# In[55]:


#Evaluation
#prediction on training data

#accuracy for prediction on training data
training_data_prediction = model.predict(X_train)


# In[56]:


print(training_data_prediction)


# In[60]:


# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load a dataset (e.g., California housing dataset)
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize the XGBRegressor model
model = XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the R squared error
score_1 = r2_score(y_test, y_pred)

# Compute the Mean Absolute Error
score_2 = mean_absolute_error(y_test, y_pred)

# Print the computed metrics
print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[61]:


#visualizing the actual and predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()


# In[62]:


#prediction on test data

# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)


# In[65]:


from sklearn import metrics

 code

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[ ]:




