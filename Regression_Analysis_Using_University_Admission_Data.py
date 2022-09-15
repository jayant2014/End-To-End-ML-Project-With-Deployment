#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Loading Dataset

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the data file and convert it to a dataframe
university_df = pd.read_csv("datasets/university_admission.csv")


# In[3]:


university_df.head()


# In[4]:


university_df.tail()


# In[5]:


# Display the feature columns
university_df.columns


# In[6]:


university_df.dtypes


# In[7]:


# Check the shape of the dataframe
university_df.shape
# Basically we have 1000 samples


# In[8]:


# Check if any missing values
university_df.isnull().sum()
# No missing values


# In[9]:


# Check the statistics
university_df.describe()


# # Exploratory Data Analysis

# In[10]:


# Check if there are any Null values
sns.heatmap(university_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()
# Basically there is no missing values hence the plot looks blank


# In[11]:


# Let's draw the histogram
university_df.hist(bins = 30, figsize = (20,20), color = 'r');
plt.show()


# In[12]:


# Let's draw a pair plot to check correlation
sns.pairplot(university_df)
plt.show()
# Here toefl score, GRE score, CGPA are more related to admission


# In[13]:


# Let's do a scatter plot for chance of admission with university rating hue
for i in university_df.columns:
    plt.figure(figsize = (13, 7))
    sns.scatterplot(x = i, y = 'Chance_of_Admission', hue = "University_Rating", hue_norm = (1,5), data = university_df)
    plt.show()


# In[14]:


# Let's plot the correlation matrix
corr_matrix = university_df.corr()
plt.figsize = (20,20)
sns.heatmap(corr_matrix, annot = True)


# # Prepare the data for training

# In[15]:


# Drop the target column
X = university_df.drop(columns = ['Chance_of_Admission'])


# In[16]:


y = university_df['Chance_of_Admission']


# In[17]:


X.shape


# In[18]:


y.shape


# In[20]:


X


# In[21]:


# Convert both numpy array
X = np.array(X)
y = np.array(y)


# In[22]:


# Reshaping the array from (1000,) to (1000, 1)
y = y.reshape(-1,1)
y.shape


# In[23]:


# Scaling the data before training the model
# Here no scaling required as we will be using XGBoost


# In[24]:


# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# # Modelling and Evaluation using XGBoost

# In[27]:


# Train an XGBoost regressor model 
import xgboost as xgb

model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 3, n_estimators = 100)
model.fit(X_train, y_train)


# In[28]:


# Predict using the testing dataset
result = model.score(X_test, y_test)
print("Accuracy : {}".format(result))


# In[29]:


# Make predictions on the test data
y_predict = model.predict(X_test)


# In[30]:


y_predict


# In[31]:


# Let's measure the performance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 


# # Model Optimization With GridSearchCV

# In[32]:


from sklearn.model_selection import GridSearchCV


# In[33]:


parameters_grid = { 'max_depth': [3, 6, 10], 
                   'learning_rate': [0.01, 0.05, 0.1],
                   'n_estimators': [100, 500, 1000],
                   'colsample_bytree': [0.3, 0.7]}


# In[34]:


model = xgb.XGBRegressor(objective ='reg:squarederror')


# In[35]:


# We use the "neg_mean_squared_error" since GridSearchCV() ranks all the algorithms (estimators) 
# and specifies which one is the best 
xgb_gridsearch = GridSearchCV(estimator = model, 
                              param_grid = parameters_grid, 
                              scoring = 'neg_mean_squared_error',  
                              cv = 5, 
                              verbose = 5)


# In[36]:


xgb_gridsearch.fit(X_train, y_train)


# In[37]:


xgb_gridsearch.best_params_


# In[38]:


xgb_gridsearch.best_estimator_


# In[39]:


y_predict = xgb_gridsearch.predict(X_test)


# In[40]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 


# In[43]:


y_predict


# In[44]:


# Scatter plot for prediction
plt.scatter(y_test, y_predict)


# In[46]:


# Residuals
residuals=y_test-y_predict
residuals


# In[54]:


X_test[1]


# In[53]:


university_df.head(1)


# In[56]:


X_test


# In[57]:


xgb_gridsearch.predict(X_test[0].reshape(1,-1))


# # Pickling the Model

# In[41]:


import pickle


# In[58]:


pickle.dump(xgb_gridsearch, open('model.pkl', 'wb'))


# In[59]:


# Load the model
pickled_model=pickle.load(open('model.pkl','rb'))


# In[60]:


pickled_model.predict(X_test[0].reshape(1,-1))


# In[ ]:




