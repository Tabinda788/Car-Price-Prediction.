import os
os.chdir('C:\\Users\\DELL\\Desktop\\nptl online course\\python for data science\\week 4')
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








































