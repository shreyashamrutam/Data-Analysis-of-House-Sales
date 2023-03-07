#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of House Sales in King County USA
# 
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. Data Analyst working at a Real Estate Investment Trust. The Trust would like to start investing in Residential real estate. You are tasked with determining the market price of a house given a set of features. You will analyze and predict housing prices using attributes or features such as square footage, number of bedrooms, number of floors, and so on

# ## Downloading the Dataset

# > Instructions for downloading the dataset (delete this cell)
# >
# > - Find an interesting dataset on this page: https://www.kaggle.com/datasets?fileType=csv
# > - The data should be in CSV format, and should contain at least 3 columns and 150 rows
# > - Download the dataset using the [`opendatasets` Python library](https://github.com/JovianML/opendatasets#opendatasets)

# In[99]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[100]:


# Change this
dataset_url = 'https://www.kaggle.com/datasets/sumaya23abdul/house-sales-in-king-county-usa' 


# In[101]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[102]:


# Change this
data_dir = './house-sales-in-king-county-usa'


# In[103]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[104]:


project_name = "house-sales-in-king-county" # change this (use lowercase letters and hyphens only)


# In[105]:


get_ipython().system('pip install jovian --upgrade -q')


# In[106]:


import jovian


# In[107]:


jovian.commit(project=project_name)


# ## Importing Data Sets
# 
# **TODO** - Loading the CSV Files and extracting them for data modeling and cleaning.
# 
# 

# > Instructions (delete this cell):
# >
# > - Load the dataset into a data frame using Pandas
# > - Explore the number of rows & columns, ranges of values etc.
# > - Handle missing, incorrect and invalid data
# > - Perform any additional steps (parsing dates, creating additional columns, merging multiple dataset etc.)

# In[108]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[109]:


pd.read_csv(data_dir + "/kc_house_data.csv")


# In[110]:


data_raw_df = pd.read_csv(data_dir + "/kc_house_data.csv")


# In[111]:


data_raw_df.head()


# In[112]:


data_raw_df.columns


# In[113]:


data_raw_df.shape


# In[114]:


data_raw_df.dtypes


# In[115]:


data_raw_df.info()


# In[116]:


data_raw_df.describe()


# In[120]:


get_ipython().system('pip install jovian --upgrade -q')


# In[121]:


import jovian


# In[122]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# **TODO** - write some explanation here.
# 
# 

# > Instructions (delete this cell)
# > 
# > - Compute the mean, sum, range and other interesting statistics for numeric columns
# > - Explore distributions of numeric columns using histograms etc.
# > - Explore relationship between columns using scatter plots, bar charts etc.
# > - Make a note of interesting insights from the exploratory analysis

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[123]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[124]:


print("Number of NaN values for the column bedrooms :", data_raw_df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bathrooms :", data_raw_df['bathrooms'].isnull().sum())


# In[125]:


mean = data_raw_df['bedrooms'].mean()
data_raw_df['bedrooms'].replace(np.nan,mean, inplace=True)
mean = data_raw_df['bathrooms'].mean()
data_raw_df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", data_raw_df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", data_raw_df['bathrooms'].isnull().sum())


# In[126]:


import jovian


# In[127]:


jovian.commit()


# ## Exploratory Data Analysis
# TODO - Using the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.
# 
# 

# #### Q1: How many no. of houses are there in unique floor values by using method.to_frame

# In[128]:


a=data_raw_df.value_counts(["floors"])
a.to_frame()


# TODO - Using the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# #### Q2: Determine and show the boxplot of whether the houses with a waterfront view or without have more price outliers 

# In[129]:


x=data_raw_df["waterfront"]
y=data_raw_df["price"]
sns.boxplot(x,y,data=data_raw_df)


# TODO - Using the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.

# #### Q3: Determine if the feature sqft_above is negatively or positively correlated with price

# In[130]:


x=data_raw_df['sqft_above']
y=data_raw_df['price']
sns.regplot(x,y,data=data_raw_df)


# We can use the Pandas method corr() to find the feature other than price that is most correlated with price.

# In[131]:


data_raw_df.corr()['price'].sort_values()


# In[132]:


import jovian


# In[133]:


jovian.commit()


# ## Data Model Development
# 
# **TODO** - We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.

# #### Q4: Determine the linear regression of price and sqft_living

# In[134]:


X = data_raw_df[['long']]
Y = data_raw_df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# Now we will be Fiting a linear regression model to predict the 'price' using the feature 'sqft_living' then calculating the R^2.

# In[135]:


X = data_raw_df[['sqft_living']]
Y = data_raw_df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# Fiting a linear regression model to predict the 'price' using the list of features:

# In[136]:


features = data_raw_df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]


# In[137]:


Y = data_raw_df['price']
lm = LinearRegression()
lm.fit(features,Y)
lm.score(features, Y)


# In[138]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[139]:


pipe=Pipeline(Input)
pipe.fit(features,y)
pipe.score(features,y)


# In[140]:


import jovian


# In[141]:


jovian.commit()


# 

# ## Data Model Evaluation and Refinement

# Now we will be importing the necessary modules:

# In[142]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("We are done witn importing the required modules for evalutation and refinement!!")


# #### Here, We will split the data into training and testing sets:
# #### Q5: Determine the total number of test samples and training samples

# In[143]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = data_raw_df[features]
Y = data_raw_df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 1)


print("Total Number of test samples :", x_test.shape[0])
print("Total Number of training samples :", x_train.shape[0])


# Now fiting in a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.

# In[144]:


from sklearn.linear_model import Ridge
print("Done with importing the requored Module!!")


# In[145]:


ridgemodel=Ridge(alpha=0.1)
ridgemodel.fit(x_train,y_train)
ridgemodel.score(x_train,y_train)


# Performing a second order polynomial transform on both the training data and testing data.creating and fiting a Ridge regression object using the training data, seting the regularisation parameter to 0.1, and calculating the R^2 utilising the test data provided.

# In[146]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(degree=2,include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x_train,y_train)
pipe.fit(x_test,y_test)
print("train : ",pipe.score(x_train,y_train))
print("test : ",pipe.score(x_test,y_test))
ridgemodel=Ridge(alpha=0.1)


# In[147]:


a=np.array(pipe.predict(x_test))
a


# In[148]:


ax1=sns.distplot(data_raw_df['price'],hist=False,color='r',label="actual")
sns.distplot(a,hist=False , color='b',label='fitted',ax=ax1)


# ## Inferences and Conclusion
# 

# #### From the following we find out the data model development for data refinement where we plot grraph of density and price of the houses which will be helpful to predicate the increase in the house rent depending upon the given requirements like floors, sqft of bathroom, view, etc. based on the choices of the buyers.

# ## References and Future Work
# 
# **TODO** - These are the following links and blogs which helped me in the project.
# 
# --1.https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/
# 
# --2.https://towardsdatascience.com/predicting-house-prices-with-machine-learning-62d5bcd0d68f
# 
# --3.https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# 
# --4.https://www.vshsolutions.com/blogs/house-price-prediction-using-regression-algorithms/#:~:text=Regression%20algorithms%2C%20on%20the%20other,is%20expected%20to%20score%20etc.
# 
# --5.https://www.section.io/engineering-education/house-price-prediction/
# 
# --6.https://www.rocketmortgage.com/learn/home-value#:~:text=One%20of%20the%20most%20accurate,you're%20preparing%20to%20sell.

# In[149]:


import jovian


# In[150]:


jovian.commit()

