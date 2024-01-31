# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:48:04 2024

@author: UTENTE
"""

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Let's first understand data

features = pd.read_csv("C:/Users/UTENTE/Desktop/walmart/features.csv.zip")
stores = pd.read_csv("C:/Users/UTENTE/Desktop/walmart/stores.csv")
test = pd.read_csv("C:/Users/UTENTE/Desktop/walmart/test.csv.zip")
train = pd.read_csv("C:/Users/UTENTE/Desktop/walmart/train.csv.zip")
sumbission_form = pd.read_csv("C:/Users/UTENTE/Desktop/walmart/sampleSubmission.csv.zip")



# Here I merged two data sets, because the stores one is basically furhter info about the features data set
features_stores = features.merge(stores, how="inner", on = "Store")

# We need to understand dates as Date
features_stores.Date = pd.to_datetime(features_stores.Date)
features_stores['Week'] = features_stores.Date.dt.isocalendar().week
features_stores['Year'] = features_stores.Date.dt.isocalendar().year
features_stores['month'] = features_stores['Date'].dt.month


test.Date = pd.to_datetime(test.Date)
train.Date = pd.to_datetime(train.Date)


features_stores.isnull().sum()

#Since Markdowns refer to some special events, the missing values means the event where not occurring
# I decided to replace Na with 0
columns_to_fill_with_zero = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]

features_stores[columns_to_fill_with_zero] = features_stores[columns_to_fill_with_zero].fillna(0)





# For the CPI and Unemployment variables we have a different nature 
# Analysing these two variables, I realised that they are missing for blocks from May to July 2013, the end of the data set
# Let's fill them with backward fill


features_stores["CPI"] = features_stores["CPI"].fillna(method="bfill")
features_stores["Unemployment"] =features_stores["Unemployment"].fillna(method="bfill")

#Now let's merge the data sets

test = test.merge(features_stores, how = "inner", on = ["Store", "Date", "IsHoliday"])
train = train.merge(features_stores, how = "inner", on = ["Store", "Date", "IsHoliday"])

test = test.drop("Date", axis = 1)
train = train.drop("Date", axis=1)


train = pd.get_dummies(train, columns=["IsHoliday", "Type"])

test = pd.get_dummies(test, columns=["IsHoliday", "Type"])

# The test set has missing values belonging to some other nature
# I performed linear interpolation, so based in Time 
test['CPI'] = test['CPI'].interpolate(method='linear')
test['Unemployment'] = test['Unemployment'].interpolate(method='linear')


test.isnull().sum()


X_train = train.drop("Weekly_Sales", axis = 1)
y_train = train["Weekly_Sales"]

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(test)


sumbission_form["Weekly_Sales"] = y_pred

