# %% [markdown]
# #  Retail Analysis with Walmart Data

# %% [markdown]
# # Data Understanding

# %% [markdown]
# This is the historical data that covers sales from 2010-02-05 to 2012-11-01, in the file Walmart_Store_sales. Within this file you will find the following fields:
# 
# - Store - the store number
# 
# - Date - the week of sales-Weekly_Sales -  sales for the given store
# 
# - Holiday_Flag - whether the week is a special holiday week 1 – Holiday week 0 – Non-holiday week
# 
# - Temperature - Temperature on the day of sale
# 
# - Fuel_Price - Cost of fuel in the region
# 
# - CPI – Prevailing consumer price index
# 
# - Unemployment - Prevailing unemployment rate

# %% [markdown]
# **Holiday Events**
# 
# Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
# 
# Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
# 
# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
# 
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
# 

# %% [markdown]
#  ## Basic Statistics Tasks To Perform
# 
# 1. Which store has maximum sales
# 
# 2. Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation
# 
# 3. Which store/s has good quarterly growth rate in Q3’2012
# 
# 4. Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in     non-holiday season for all stores together
# 
# 5. Provide a monthly and semester view of sales in units and give insights

# %% [markdown]
# **Q1**. Which store has maximum sales
#    
# Solution :

# %%
# First Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# Read Data from CSV file and store in df Variable
df=pd.read_csv('Walmart_Store_sales.csv')
df.head()

# %% [markdown]
# # Data Prepartion

# %%
#To Check shape of Data
df.shape

# %%
# To check is there any null value in data
df.isna().sum()

# %% [markdown]
# **Observation :** There is No Null Values

# %%
df.info()

# %% [markdown]
# **Observation :** Above datatype show that Date  is Object type we need to change it to Date dtype

# %%
# Changing Object Type to Date Type
df['Date'] = pd.to_datetime(df['Date'])
df.info()

# %%
#To Check Basic Statistics
df.describe()

# %%
# Splitting Date and create new columns ( Day,Month and Year) as it question number 5
df["Day"]= pd.DatetimeIndex(df['Date']).day
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Year'] = pd.DatetimeIndex(df['Date']).year
df

# %%
# To Get All Total Unique  Store Number
Total_Unique=df['Store'].unique()
print("Number of Unique Store are",Total_Unique)

# %% [markdown]
# **Observation -** We have 45 Store

# %% [markdown]
# # Task to Perform

# %% [markdown]
# **Q1**. Which store has maximum sales

# %%
#To Get Maximum Sales values
df.sort_values(by='Weekly_Sales',ascending=False).groupby('Store')['Weekly_Sales'].sum().round().max()

# %%
#To Get Highest Store Name 
highest_Sales=df.groupby('Store')['Weekly_Sales'].sum()
highest_Sales.idxmax()

# %% [markdown]
# **Observation** :
#  - By Looking at above  Calculation we can say that 20 Store has maximum sales 
#  - Lets Confirm by Visulation

# %%
# Plot and Check the 20 Store has maximum sales or Not
plt.figure(figsize = (15,8))
ax = sns.barplot(x="Store", y="Weekly_Sales", ci=None,data=df)

# %% [markdown]
# **Observation** :
#  - By Looking at Above bar Plot we can justify that 20 Store has maximum sales 

# %% [markdown]
# **Q2**. Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation

# %%
df.describe()

# %% [markdown]
# **Formula** :
#  Coefficient of Variation (CV) = (Standard Deviation/Mean) × 100

# %%
# #To Get Highest Standard Deviation
Highest_Standard_Deviation=df.groupby('Store')['Weekly_Sales'].std()
print("Highest Standard Deviation of Sales Store :",Highest_Standard_Deviation.idxmax())

# %%
df_store_14 =df.loc[df['Store']==14] # To Select Store 14 in Another DataFrame
df_store_14.describe() # To Get Basic Statitics Stat Like Standard Devivation and Mean in Store 14

# %% [markdown]
# **Observation :**  We get Standard Deviation = 	3.175699e+05 & Mean =2.020978e+06	  

# %%
std_dev=3.175699e+05	# Store in Standard Deviation in std_dev of 14 Store
mean=2.020978e+06	    # Store in Mean in Mean of 14 Store
CV = (std_dev / mean)*100 # Calculating Coefficient of Variation  of 14 Store
print("Coefficient of Variation of Store 14 =",CV)

# %% [markdown]
# **Observation** :
#  - By Looking at by Calculation we can justify that 14 Store has Maximum sales and its Coefficient of Variation  is 15.71367427057593

# %% [markdown]
# **Q3**. Which store/s has good quarterly growth rate in Q3’2012

# %%
# Third Quater Which range from 7 - 10 Month
walmart_data_Q32012 = df[((df['Date']) >= '2012-07-01') & ((df['Date']) <= ('2012-09-01'))]
walmart_data_Q32012 

# %%
walmart_data_growth = walmart_data_Q32012.groupby(['Store'])['Weekly_Sales'].sum() #To get data growth  
walmart_data_growth

# %%
print("Store Number", walmart_data_growth.idxmax(),"has Good Quartely Growth in Q3'2012 with Values :",walmart_data_growth.max())

# %%
plt.figure(figsize = (15,8))
plotting_Highest_Sales = sns.barplot(x="Store", y="Weekly_Sales", ci=None,data=walmart_data_Q32012)

# %% [markdown]
# 
# 
#  **Observation** 
#  - By looking at Above Barplot we can say that Store 4  has Good Quartely Growth in Q3'2012
#   

# %% [markdown]
# **Q4**. Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in   non-holiday season for all stores together

# %% [markdown]
# **Holiday Event :**

# %% [markdown]
# Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13 [2010-02-12, 2011-02-11,2012-02-10,2013-02-08]
# 
# Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13 [2010-09-10,2011-09-09,2012-09-07,2013-09-06]
# 
# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13[2010-11-26,2011-11-25,2013-11-23,2013-11-29]
# 
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13[2010-12-31, 2011-12-30, 2012-12-28, 2013-12-27]
# 

# %%
stores_holiday_sales = df[df['Holiday_Flag'] == 1] # Store of Holiday_Sales
stores_holiday_sales

# %%
stores_nonholiday_sales = df[df['Holiday_Flag'] == 0] # Store of Non_Holiday_Sales
stores_nonholiday_sales

# %% [markdown]
# **Holiday_Event -**

# %%
#Stores Sales in Super Bowl Day
#Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
stores_holiday_sales_superBowl=stores_holiday_sales[((stores_holiday_sales['Date']) == '12-02-2010')|
                                                      ((stores_holiday_sales['Date']) == '11-02-2011')|
                                                      ((stores_holiday_sales['Date']) == '10-02-2012')|
                                                      ((stores_holiday_sales['Date']) == '08-02-2013')]

# %%
#Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13 

stores_holiday_sales_labourDay=stores_holiday_sales[((stores_holiday_sales['Date']) == '10-09-2010')|
                                                      ((stores_holiday_sales['Date']) == '09-09-2011')|
                                                      ((stores_holiday_sales['Date']) == '07-09-2012')|
                                                      ((stores_holiday_sales['Date']) == '06-09-2013')]

# %%
#Stores Sales in Thanks Giving
#Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
stores_holiday_sales_thanksgiving =stores_holiday_sales[((stores_holiday_sales['Date']) == '26-11-2010')|
                                                       ((stores_holiday_sales['Date']) == '25-11-2011')|
                                                       ((stores_holiday_sales['Date']) == '23-11-2012')|
                                                       ((stores_holiday_sales['Date']) == '29-11-2013')]


# %%
#Stores Sales in Christmas
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
stores_holiday_sales_Christmas =stores_holiday_sales[((stores_holiday_sales['Date']) == '31-12-2010')|
                                                      ((stores_holiday_sales['Date']) == '30-12-2011')|
                                                      ((stores_holiday_sales['Date']) == '28-12-2012')|
                                                      ((stores_holiday_sales['Date']) == '27-12-2013')]

# %%
print("Super Bowl Day Sale",stores_holiday_sales_superBowl['Weekly_Sales'].mean().round(2))
print("Labour Day Sale",stores_holiday_sales_labourDay['Weekly_Sales'].mean().round(2))
print("Thanksgiving Day Sale",stores_holiday_sales_thanksgiving['Weekly_Sales'].mean().round(2))
print("Christmas Day Sale",stores_holiday_sales_Christmas['Weekly_Sales'].mean().round(2))

# %% [markdown]
# **Observation :** Thanksgiving Day Sale has Highest Mean in Holiday Sale

# %% [markdown]
# **Non-Holiday Sales :**

# %%
# Mean sales in the non-holiday season for all stores together.
non_holiday_sales = df[(df['Holiday_Flag'] == 0)]['Weekly_Sales'].mean().round(2)
print("Mean sales in the non-holiday season for all stores together :\n",non_holiday_sales)

# %% [markdown]
# **Observation**
# - We found that Thanksgiving has the highest sales 1,471,273.43 than non-holiday sales 1,041,256.38

# %% [markdown]
# **Q5**. Provide a monthly and semester view of sales in units and give insights

# %%
df.head()

# %%
plt.figure(figsize=(14,8))
Month_View = sns.barplot(x="Month", y="Weekly_Sales", data=df,ci=None)
Month_View.set_title('Monthly View Of Sales')

# %% [markdown]
# **Observation :** 
#   - Above Bar plot Show the  December has Highest Sales

# %%
# Yearly view of sales
plt.figure(figsize=(10,6))

df.groupby("Year")[["Weekly_Sales"]].sum().plot(kind='bar',legend=False)
plt.xlabel("years")
plt.ylabel("Weekly Sales")
plt.title("Yearly view of sales");

# %% [markdown]
# **Observation-**
# I have drawn some insights 
#  1. Year 2010 has the highest sales and 2012 has the lowest sales.
#  2.  December month has the highest weekly sales.
#  3.  Year 2011 has the highest weekly sales
# 
# 
# 

# %% [markdown]
# # Statistical Model

# %% [markdown]
# For Store 1 – Build  prediction models to forecast demand.Linear Regression – Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order). Hypothesize if CPI, unemployment, and fuel price have any impact on sales.
# 
# Change dates into days by creating new variable.
# 
# Select the model which gives best accuracy

# %% [markdown]
# - For Store 1 – Build  prediction models to forecast demand. Linear Regression – Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order). Hypothesize if CPI, unemployment, and fuel price have any impact on sales.
# 

# %% [markdown]
# **Before Buliding Model find and remove  outliers**

# %%
# Before Buliding Model find and remove  outliers 
fig, axs = plt.subplots(4,figsize=(6,18))
X = df[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(df[column], ax=axs[i])

# %% [markdown]
# **Observation :** 
#  -  Unemployment and Temperature has Outlier

# %%
# drop the outliers     
data_new = df[(df['Unemployment']<10) & (df['Unemployment']>4.5) & (df['Temperature']>10)]
data_new

# %%
# Before Buliding Model find and remove  outliers 
fig, axs = plt.subplots(4,figsize=(6,18))
X = data_new[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(data_new[column], ax=axs[i])

# %%
# Import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# %%
df_Store_1=data_new.loc[data_new['Store'] == 1]

# %%
df_Store_1.head()

# %% [markdown]
# - Checking attribue of CPI, unemployment, and fuel price have any impact on sales.

# %%
# Checking attribue of Unemployment vs  Weekly_Sales
x = df_Store_1['Unemployment'] 
y = df_Store_1['Weekly_Sales']

plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)# r should be between -1 to 1
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)

plt.plot(x, mymodel)
plt.show()

# %%
# Checking attribue of CPI vs  Weekly_Sales
x = df_Store_1['CPI'] 
y = df_Store_1['Weekly_Sales']

plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)# r should be between -1 to 1
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)

plt.plot(x, mymodel)
plt.show()

# %%
# Checking attribue of Fuel_Price vs  Weekly_Sales
x = df_Store_1['Fuel_Price'] 
y = df_Store_1['Weekly_Sales']

plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)# r should be between -1 to 1
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)

plt.plot(x, mymodel)
plt.show()

# %%
# Checking attribue of Temperature vs  Weekly_Sales
x = df_Store_1['Temperature'] 
y = df_Store_1['Weekly_Sales']

plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)# r should be between -1 to 1
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)

plt.plot(x, mymodel)
plt.show()

# %% [markdown]
# 
#  Q.  Change dates into days by creating new variable.
# 
# 
# Ans : We Have already change in Data Prepartion Process 

# %%
# Just For Checking 
data_new.head()

# %%
# Select features and target 
X = data_new[['Store','Fuel_Price','CPI','Unemployment','Day','Month','Year']]
y = data_new['Weekly_Sales']

# Split data to train and test (0.80:0.20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# %% [markdown]
# 
#  -  Select the model which gives best accuracy

# %%
# Linear Regression model
print('Linear Regression:')
print()
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Accuracy:',reg.score(X_train, y_train)*100)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


sns.scatterplot(y_pred, y_test);

# %% [markdown]
# 
#  - We Have See Less Accuracy in Linear Regression

# %%
# Random Forest Regressor
print('Random Forest Regressor:')
print()
rfr = RandomForestRegressor(n_estimators = 400,max_depth=15,n_jobs=5)        
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print('Accuracy:',rfr.score(X_test, y_test)*100)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


sns.scatterplot(y_pred, y_test);

# %% [markdown]
# - We Have See Great Accuracy in Random Forest Regressor with Accuracy: 94.57900803858068
#  

# %% [markdown]
# **Observation :**
#  - We select Random Forest Regressor as it has better Accuracy than Linear Regression 


