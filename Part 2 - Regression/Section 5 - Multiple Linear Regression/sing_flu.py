# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Importing the dataset_polyclinic
dataset_polyclinic_org = pd.read_csv('average-daily-polyclinic-attendances-for-selected-diseases.csv')
dataset_polyclinic = pd.DataFrame(columns=['epi_week', 'disease', 'no._of_cases'])
for n in range (len(dataset_polyclinic_org)):
    if dataset_polyclinic_org.iloc[n,1:2].values == 'Acute Upper Respiratory Tract infections':
        dataset_polyclinic = dataset_polyclinic.append(dataset_polyclinic_org.loc[n])
dataset_polyclinic['epi_week_edited'] = dataset_polyclinic['epi_week'].str.replace('-W','').astype(int)
dataset_polyclinic['LastDayWeek'] = pd.to_datetime((dataset_polyclinic['epi_week_edited']-1).astype(str) + "6", format="%Y%U%w")
dataset_polyclinic = dataset_polyclinic[dataset_polyclinic['LastDayWeek']>datetime.datetime(2011,12,31)]
dataset_polyclinic = dataset_polyclinic[dataset_polyclinic['LastDayWeek']<datetime.datetime(2017,1,1)]
dataset_polyclinic['YearMax'] = pd.DatetimeIndex(dataset_polyclinic['LastDayWeek']).year
dataset_polyclinic['MonthMax'] = pd.DatetimeIndex(dataset_polyclinic['LastDayWeek']).month
dataset_polyclinic['MonthMax'] =  dataset_polyclinic['MonthMax'].apply(lambda x: '{0:0>2}'.format(x))
dataset_polyclinic['month'] = dataset_polyclinic['YearMax'].astype(str) + '-' + dataset_polyclinic['MonthMax'].astype(str)
dataset_polyclinic = dataset_polyclinic.drop('epi_week', axis=1)
dataset_polyclinic = dataset_polyclinic.drop('epi_week_edited', axis=1)
dataset_polyclinic = dataset_polyclinic.drop('LastDayWeek', axis=1)
dataset_polyclinic = dataset_polyclinic.drop('YearMax', axis=1)
dataset_polyclinic = dataset_polyclinic.drop('MonthMax', axis=1)
dataset_polyclinic = dataset_polyclinic.groupby(['month'])['no._of_cases'].sum().reset_index()
dataset_polyclinic = dataset_polyclinic.sort_values('month', ascending=True)

# Import the dataset_humidity
dataset_humidity = pd.read_csv('relative-humidity-monthly-mean.csv')
dataset_humidity['month_edited'] = dataset_humidity['month'].str.replace('-','').astype(int)
dataset_humidity['Date'] = pd.to_datetime((dataset_humidity['month_edited']).astype(str), format="%Y%m")
dataset_humidity = dataset_humidity[dataset_humidity['Date']>datetime.datetime(2011,12,31)]
dataset_humidity = dataset_humidity[dataset_humidity['Date']<datetime.datetime(2017,1,1)]
dataset_humidity = dataset_humidity.drop('month_edited', axis=1)
dataset_humidity = dataset_humidity.drop('Date', axis=1)
dataset_humidity = dataset_humidity.sort_values('month', ascending=True).reset_index()
dataset_humidity = dataset_humidity.drop('index', axis=1)
dataset_humidity = dataset_humidity.drop('month', axis=1)

dataset = pd.concat([dataset_polyclinic, dataset_humidity], axis=1, join_axes=[dataset_polyclinic.index])

dataset_temperature = pd.read_csv('surface-air-temperature-monthly-mean.csv')
dataset_temperature['month_edited'] = dataset_temperature['month'].str.replace('-','').astype(int)
dataset_temperature['Date'] = pd.to_datetime((dataset_temperature['month_edited']).astype(str), format="%Y%m")
dataset_temperature = dataset_temperature[dataset_temperature['Date']>datetime.datetime(2011,12,31)]
dataset_temperature = dataset_temperature[dataset_temperature['Date']<datetime.datetime(2017,1,1)]
dataset_temperature = dataset_temperature.drop('month_edited', axis=1)
dataset_temperature = dataset_temperature.drop('Date', axis=1)
dataset_temperature = dataset_temperature.sort_values('month', ascending=True).reset_index()
dataset_temperature = dataset_temperature.drop('index', axis=1)
dataset_temperature = dataset_temperature.drop('month', axis=1)

#dataset = pd.concat([dataset, dataset_temperature], axis=1, join_axes=[dataset.index])

dataset_rain = pd.read_csv('rainfall-monthly-total.csv')
dataset_rain['month_edited'] = dataset_rain['month'].str.replace('-','').astype(int)
dataset_rain['Date'] = pd.to_datetime((dataset_rain['month_edited']).astype(str), format="%Y%m")
dataset_rain = dataset_rain[dataset_rain['Date']>datetime.datetime(2011,12,31)]
dataset_rain = dataset_rain[dataset_rain['Date']<datetime.datetime(2017,1,1)]
dataset_rain = dataset_rain.drop('month_edited', axis=1)
dataset_rain = dataset_rain.drop('Date', axis=1)
dataset_rain = dataset_rain.sort_values('month', ascending=True).reset_index()
dataset_rain = dataset_rain.drop('index', axis=1)
dataset_rain = dataset_rain.drop('month', axis=1)

dataset = pd.concat([dataset, dataset_rain], axis=1, join_axes=[dataset.index])

X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, 1].values
          
X2=X*X
X3=X*X*X
X = np.concatenate((X,X2,X3), axis=1) 
X = X[:,(0,2,4,5)]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#fitting to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#prediction test set result
y_pred = regressor.predict(X_test)              

plt.plot(y_pred,label='y_pred')
plt.plot(y_test,label='y_test')
plt.legend()
plt.show()

#Optimal modle by backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((60,1)).astype(int),values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

