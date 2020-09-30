import os

#Predicting price of pre-owned cars
import pandas as pd
import numpy as np
import seaborn as sns

# Setting dimensions as plot
 sns.set(rc={'figure.figsize':(11.7,8.27)})

#Reading csv file
 cars_data = pd.read_csv('cars_sampled.csv')

#Copy data
 cars=cars_data.copy()

#Structure of data
 cars.info()
#Summarizing data
 cars.describe()
 pd.set_option('display.float_format',lambda x: '%.3f' % x)
cars.describe()
#To display maximum set of columns
pd.set_option('display.max_columns',500)
cars.describe()

#Dropping unwanted columns
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)

#Removing duplicate records
cars.drop_duplicates(keep='first',inplace=True)
#470 duplicate records

#Number of missing values in each column
cars.isnull().sum()

#Variable year of registration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)

# Working range -1950 to 2018

#Variable Price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#working range 100 to 150000

#Variable PowerPS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)

sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#working range 10 and 500
#Working rannge of data
cars=cars[
         (cars.yearOfRegistration <= 2018)
       & (cars.yearOfRegistration >= 1950)
       & (cars.price >= 100)
       & (cars.price <= 150000)
       & (cars.powerPS >= 10)
       & (cars.powerPS <= 500)]
#6700 records are dropped
#Furthur to simplify - variablle reduction
#Combining year of registration and month of registration

cars['monthOfRegistration']/=12

#Creating new variable age by adding year of registration and month of registration

cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

#Dropping year of registration and month of registration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualising parameters

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#visualizing parameters after narrowing working ranges
#Age vs Price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)

#Cars price higher are fever
#with increase in age price decreases
#However some cars are priced higher with increase in age


#powerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)


#Variable Seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#Fewer cars have commercial => insignficant

#Variable offertype
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)

#Variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
#Equally distributed
sns.boxplot(x='abtest',y='price',data=cars)
#For every price value there is almost 50-50 distribuion
#Does not affect price => insignificant

#Variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

#Variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#gearbox affects price

#Variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)

#Variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',data=cars,fit_reg=False)
#Considered in modelling

#Variable fuelType
cars['fuelType'].value_counts().sort_index()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#Fueltype affects price

#Variable brand
cars['brand'].value_counts().sort_index()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#cars are distributed over various brands
#Considered for modelling

#Variable notRepairedDamage
#yes-car is damaged but not rectified
#no-car was damaged but has been rectified
cars['notRepairedDamage'].value_counts().sort_index()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
#As expected the cars that require the damages to be repaired
#fall under low price ranges

#Removing insignificant variables
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#Correlation
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

"""
We are going to build a linear regresssion and random forest model
on 2 sets of data
1. Data obtained by ommiting rows with missing values
2. Data obtained by imputing missing values
"""

cars_omit=cars.dropna(axis=0)

#Converting catagorical variables into dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

#Importing neessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Building model with omitted data
#Separating input and output features

x1 = cars_omit.drop(['price'],axis='columns',inplace=False)
y1 = cars_omit['price']

#plotting the variable price
prices = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()

#Transforming price as logrithmic value
y1=np.log(y1)

#splitting data into test and train
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#Baseline model for ommited data

"""
We are making a baseline model by using test data mean value
This is to set a bench mark and to compare with our regression model
"""

#Fitting the mean for test data value
base_pred=np.mean(y_test)
print(base_pred)

#Repeating same value till length of test data
base_pred=np.repeat(base_pred,len(y_test))

#Finding the root mean square error
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

#Linear Regression with ommited data
#Setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#Model
Model_lin1=lgr.fit(X_train,y_train)

#Predicting model on test data
cars_prediction_lin1=lgr.predict(X_test)

#Computing MSE and RMSE
lin_mse1=mean_squared_error(y_test,cars_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R-squared vale
r2_lin_test1=Model_lin1.score(X_test,y_test)
r2_lin_train1=Model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagnosis-Residual plot analysis
residuals1=y_test-cars_prediction_lin1
sns.regplot(x=cars_prediction_lin1,y=residuals1,scatter=True,fit_reg=False)
residuals1.describe()
#Ramdom Forest with ommited data
rf = RandomForestRegressor(n_estimators=100,max_features='auto',
                          max_depth=100,min_samples_split=10, 
                          min_samples_leaf=4,random_state=1)



#Model
model_rf1=rf.fit(X_train,y_train)
#Predicting model on test set
cars_prediction_rf1=rf.predict(X_test)

#Computing MSE and RMSE
rf_mse1=mean_squared_error(y_test,cars_prediction_rf1)
lin_rmse1=np.sqrt(rf_mse1)
print(lin_rmse1)

#Rsquared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#Model building with imputed data
cars_imputed=cars.apply(lambda x:x.fillna(x.median())\
                        if x.dtype == 'float' else \
                        x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)

#Model building with imputed data

#Separating input and output features

x2 = cars_imputed.drop(['price'],axis='columns',inplace=False)
y2 = cars_imputed['price']

#plotting the variable price
prices = pd.DataFrame({"1. Before":y2, "2. After":np.log(y2)})
prices.hist()

#Transforming price as logrithmic value
y2=np.log(y2)

#splitting data into test and train
X_train1,X_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#Baseline model for imputed data

"""
We are making a baseline model by using test data mean value
This is to set a bench mark and to compare with our regression model
"""

#Fitting the mean for test data value
base_pred=np.mean(y_test1)
print(base_pred)

#Repeating same value till length of test data
base_pred=np.repeat(base_pred,len(y_test1))
 
#Finding the root mean square error
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test1,base_pred))
print(base_root_mean_square_error)

#Linear Regression with imputed data
#Setting intercept as true
lgr2=LinearRegression(fit_intercept=True)

#Model
Model_lin2=lgr2.fit(X_train1,y_train1)

#Predicting model on test data
cars_prediction_lin2=lgr2.predict(X_test1)

#Computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1,cars_prediction_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

#R-squared vale
r2_lin_test2=Model_lin2.score(X_test1,y_test1)
r2_lin_train2=Model_lin2.score(X_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

#Ramdom Forest with ommited data
rf2 = RandomForestRegressor(n_estimators=100,max_features='auto',
                          max_depth=100,min_samples_split=10, 
                          min_samples_leaf=4,random_state=1)



#Model
model_rf2=rf2.fit(X_train1,y_train1)
#Predicting model on test set
cars_prediction_rf2=rf2.predict(X_test1)

#Computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1,cars_prediction_rf2)
lin_rmse2=np.sqrt(rf_mse2)
print(lin_rmse2)

#Rsquared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)


#Final Output
print("Metrics from models built on data where missing values are ommited")

print("R square value for train for linear regression=%s"% r2_lin_train1)
print("R square value for test for linear regression=%s"% r2_lin_test1)
print("R square value for train for Random Forest=%s"% r2_rf_train1)
print("R square value for test for Random Forest=%s"% r2_rf_test1)
print("Base RMS built on data with missing values=%s"% base_root_mean_square_error)
print("Rms value for test for linear regression=%s"% lin_rmse1)

print("\n\n")

print("Metrics from models built on data where imputed values are ommited")

print("R square value for train for linear regression=%s"% r2_lin_train2)
print("R square value for test for linear regression=%s"% r2_lin_test2)
print("R square value for train for Random Forest=%s"% r2_rf_train2)
print("R square value for test for Random Forest=%s"% r2_rf_test2)
print("Base RMS built on data with imputed values=%s"% base_root_mean_square_error)
print("Rms value for test for linear regression=%s"% lin_rmse2)








































