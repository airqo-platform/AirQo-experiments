#!/usr/bin/env python
# coding: utf-8

# # Description of what the model currently does
# 
# 
# ## Forecasting
# 
# THis would be updated hourly
# 
# THis takes in use inputs of date and device (in the future this will be gps driven)
# The channel selects the best configuration which has been identified in the dict based on the training model
# 
# The sarima model then runs on the historical data up to the previous hour and using the best model to generate a 24 hour prediction. The model at this stage also includes a 95% confidence interval limits.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import datetime
import time 

import psutil
import ast


# In[ ]:





# In[4]:


sns.set_style("white", {'axes.grid' : False})
sns.set_palette("tab20", 16)
sns.set(font="Calibri")
sns.set_context('talk')


#  List of available channels:
#         
# Channel:  aq_04
# Channel:  aq_05
# Channel:  aq_06
# Channel:  aq_07
# Channel:  aq_08
# Channel:  aq_09
# Channel:  aq_10
# Channel:  aq_11
# Channel:  aq_13
# Channel:  aq_14
# Channel:  aq_15
# Channel:  aq_16
# Channel:  aq_17
# Channel:  aq_18
# Channel:  aq_20
# Channel:  aq_21
# Channel:  aq_22
# Channel:  aq_23
# Channel:  aq_24
# Channel:  aq_25
# Channel:  aq_26
# Channel:  aq_27
# Channel:  aq_29
# Channel:  aq_30
# Channel:  aq_31
# Channel:  aq_32
# Channel:  aq_33
# Channel:  aq_34
# Channel:  aq_35
# Channel:  aq_36
# Channel:  aq_39
# Channel:  aq_40
# Channel:  aq_41
# Channel:  aq_43
# Channel:  aq_44
# Channel:  aq_45
# Channel:  aq_46
# Channel:  aq_47
# Channel:  aq_48
# Channel:  aq_49

# In[5]:


hourly_data = pd.read_csv('hourly_data_AUG_19.csv', parse_dates=['time'])


# In[6]:


hourly_data


# # Forecasting

# In[7]:


best_config_dict = {'aq_22': [(2, 0, 24), (1, 0, 1, 24), 'c'] , 'aq_23': [(2, 0, 24), (0, 0, 1, 24), 'c'], 'aq_24': [(5, 0, 24), (0, 0, 0, 24), 'c'] }


# In[8]:


enter_time = input("Enter a valid date and time (yyyy/mm/dd/hh)   ")


# In[10]:


enter_chan = input("Enter a valid channel name: ")


# In[11]:


current_best_config = best_config_dict.get(str(enter_chan))
print(current_best_config)


# In[12]:


# Wrnagling the user input date formats to get a datetime
enter_time_split = enter_time.split("/")
enter_time_tuple = tuple([int(i) for i in enter_time_split])
endings = (0,0,0,0,0,)
start_time = (enter_time_tuple + endings)
start_pred_time = datetime.datetime.fromtimestamp(time.mktime(start_time))
print(start_pred_time)


# In[13]:


# interpolating, removing nans and dropping columns
def fill_gaps_and_set_datetime(d):
    # Interpolating gaps within the data
    d = d.set_index('time')
    d = d.drop('channel', axis=1)
    d_cleaned = d.interpolate(method='time');
    return d_cleaned


# In[14]:


# sarima forecast
# Takes the history (train set plus day by day testing) and configuration
# converts history values to a single long series
# generates the sarima model based on config parameters
# fits the sarima model to the series data
# creates yhat, a prediction of the next 24 hours int he test set
def sarima_forecast(history, config):
    order, sorder, trend = config
    # convert history into a univariate series
    series = to_series(history)
    # define model
    model = SARIMAX(series, order=order, seasonal_order=sorder, trend = trend,enforce_stationarity=False, enforce_invertibility=False)
#     model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast using predict method
#     yhat = model_fit.predict(len(series), len(series)+23)
    # This does th same thing but using forecast which has slightly different features
    fcst = model_fit.get_forecast(24)
    # Generating list of mean forecasts
    yhat = fcst.predicted_mean
    # Generating confidence intervals
    yhat_ci = fcst.conf_int()
    return yhat, yhat_ci


# In[15]:


# convert windows of weekly multivariate data into a series of total power
def to_series(data):
    # extract just the total power from each week
    series = [day for day in data]
    # flatten into a single series
    series = array(series).flatten()
#     print('series.shape: ', series.shape)
    return series


# In[16]:


def new_forecast(enter_chan, enter_time):

    current_best_config = best_config_dict.get(str(enter_chan))
    # converting the start prediction time to datetime
    start_pred = (pd.to_datetime(start_pred_time))

    # select all data relating to a particular channel
    all_channel_data = hourly_data.loc[hourly_data.channel == enter_chan]

    # clean the channel data, fill gaps and set datetime as index
    all_channel_data_clean = fill_gaps_and_set_datetime(all_channel_data)

    # set start time as being 8 weeks before start of forecast
    # THis will need to be changed to max number of whole days

    ####TODO #######
    start_base_time = str(start_pred - datetime.timedelta(84)) +'+3:00'
    start_pred = str(start_pred) +'+3:00'
    print((start_base_time))
    print((start_pred))
    # select final range of data to be used in forecast
    data_to_use = all_channel_data_clean.loc[start_base_time:(start_pred)].values

    yhat, yhat_ci = sarima_forecast(data_to_use, current_best_config)

    return start_pred, yhat, yhat_ci


# In[17]:


start_pred, yhat, yhat_ci = new_forecast(enter_chan, enter_time)


# In[18]:


yhat


# In[19]:



forecast = pd.DataFrame(list(zip(yhat, yhat_ci)))
forecast.columns = ['yhat', 'yhat_ci']
# Splitting confidence intercals into upper and lower bounds
forecast['yhat_lower_ci'] = [pair[0] for pair in forecast.yhat_ci]
forecast['yhat_upper_ci']  = [pair[1] for pair in forecast.yhat_ci]
# adjusting column names and dropping combinded ci
forecast.columns = ['yhat', 'yhat_ci','yhat_lower_ci', 'yhat_upper_ci']
forecast = forecast.drop('yhat_ci', axis=1);

# Generating dataframe including the actual hours predicted
start_pred_hour = pd.to_datetime(start_pred).hour +1
pred_hours = [(x + start_pred_hour)%24 for x in np.arange(24)]
forecast['pred_hours'] = pred_hours

forecast


# In[20]:


# Plotting forecasts and confidence intervals - why are they fixed?
fig, ax = plt.subplots(figsize=(20,4))
sns.lineplot(data= forecast.iloc[:, :3])
hours_order = list(forecast.pred_hours)
print(hours_order)
plt.fill_between(forecast.index,forecast.yhat_lower_ci, forecast.yhat_upper_ci, alpha=0.5)
plt.xlim(0,23)
# ax.set_xticklabels(labels=hours_order)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:





# In[7]:


ranger = np.arange(24)


# In[9]:


ranger[2:6]


# In[ ]:




