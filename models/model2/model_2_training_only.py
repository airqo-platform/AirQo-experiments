#!/usr/bin/env python
# coding: utf-8

# # Description of what the model currently does
# 
# 
# ## Model training
# Takes data for each channel and resamples into hourly data
# Interpolates to fill the gaps - using linear interpolation
# Deletes any nans at the start of the rows
# Ensures the dataset contains only full days ie day 1 starts at 0.00 and day n finishes at 2300
# Generates a list of all channels
# If chosen prints out ACF and APAXF plots for each (would be good to have this as data rather than just a plot so we can use it)
# 
# Dataset split into final test (24 hour out of sample test set), 7 day/168 hour validation set and all that comes before training set
# 
# For each channel 'flatten' the training set into a series of 24 X no of days known as 'history'
# create a config list - in this case it is made up of the different combinations in sarima configs ie order,sorder and trend
# 
# Using the sarmia params function generate a list of all configurations to be attempted
# 
# Run a grid search using the model, the train and test data and the configuration
# 
# Run through the Score model function. This tries the evaluate_model using each configuration of sarima config to generate a result. If there are any errors it ignores but if value is generated it restates the model and its score for that channel
# 
# Evaluate model function this takes history which is equal to the training set ie a number of lists each 24 hours in length. It then flattens the array to create a long list equal to history. Then for each of the 7 days in the test set it trains the model on the history data and the current configuration and generates a prediction of 24 hours. THis is added to the predctions list. Finally day i is added to history and included in the data that is used to train day i+1. This continues through each of the 7 days. At the end of each day a set of 24 scores is compared with the actual value for the next day. These are listed as SCORES.
# 
# Teh evaluate model then takes the two 7 X 74 arrays ie actual and prediction and calculates the mse/rmse of the whole by taking the root of the mean squared difference between the two. THis gives SCORE
# 
# Finally sort the configurations and scores in order with lowest at the top, print the first five. 
# Best config is the top ie the configuration with the lowest score over the test array
# This is the model to be used.
# A dict is created shwoing the optimal configuration for each channel
# 

# In[1]:


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





# In[2]:


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

# # Running the  model for each dataset

# # FUNCTIONS

# In[3]:


hourly_data = pd.read_csv('hourly_data_AUG_19.csv', parse_dates=['time'])


# In[4]:


hourly_data


# In[5]:


def full_days_only(d):
    ###  Interpolating gaps within the data and presenting in necessary format ####
    d = d.set_index('time')
    d = d.interpolate(method='time');
#     print(d)
    # drop any remaining nans from the start
    d =  d.dropna().reset_index();
    # Setting first complete day and last complete day to assist with training quality
    first_full_day = pd.to_datetime(d.loc[d.time.dt.hour == 0.00, 'time'][:1].values[0], utc=True)
    last_full_day = pd.to_datetime(d.loc[d.time.dt.hour == 23.00, 'time'][-1:].values[0], utc=True)  
    # THen correct Kampala values are applied
    d_whole_days = d.loc[(d.time >= first_full_day) & (d.time <= last_full_day)]
    d_whole_days = d_whole_days.set_index('time')
#     print(d_whole_days)
    return d_whole_days


# In[6]:


# THis should be unecessary but every time i take it out it causes issues
def out_of_sample(d, oos_size):
    df = d.iloc[:-oos_size, 1]
    oos = d.iloc[-oos_size:, 1]
    return df, oos


# In[7]:


# Creating a list of channels to iterate through
chanlist = hourly_data.channel.unique().tolist()


# In[8]:


def plot_acf_pcf(df, lags):
    # plots specifying number of lags to includ
    plt.figure(figsize=(20,6))
    # acf
    axis = plt.subplot(2, 1, 1)
    plot_acf(df, ax=axis, lags=lags)
    # pacf
    axis = plt.subplot(2, 1, 2)
    plot_pacf(df, ax=axis, lags=lags)
    # show plot
#     print(chan)
    plt.show()


# In[9]:


for chan in chanlist:
    d = hourly_data.loc[hourly_data.channel == chan, ['time','pm_2_5']]
#     print(d)
    df = full_days_only(d)
#     print('Channel: ', chan)
    # USE THIS IF WANT TO GENERATE AUTOCORRELATION GRAPHS
#     plot_acf_pcf(df, 24)


# # Preparing data for forecast model

# In[10]:


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks with 1 week validation and 1 day oos test
    train, test = data[0:-192], data[-192:-24]
    final_test = data[-24:]
    # restructure into windows of weekly data
    train = array(split(train, (len(train)/24)))
    test = array(split(test, (len(test)/24)))
#     print(train, test, final_test)
    return train, test, final_test


# In[11]:


# convert windows of weekly multivariate data into a series of total power
def to_series(data):
    # extract just the total power from each week
    series = [day for day in data]
    # flatten into a single series
    series = array(series).flatten()
#     print('series.shape: ', series.shape)
    return series


# In[12]:


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
    # make one step forecast
    yhat = model_fit.predict(len(series), len(series)+23)
    return yhat


# In[13]:


# evaluate a single model which creates a prediction for ech day and each hour
# This is then fed into the evaluate forecast function to generate overall scores for the model
# for the model
# This needs to happen for every incarnation of the model
def evaluate_model(model_func, train, test, config):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week

        yhat_sequence = model_func(history, config)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    # evaluate predictions days for each week
    score, scores = evaluate_forecasts(test, predictions)
    return score, scores


# In[14]:


def evaluate_forecasts(actual, predicted):
#     print('actual.shape : ', actual.shape)
#     print('predicted.shape', predicted.shape)
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        print('i', i)
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        print('mse', mse)
        # calculate rmse
        rmse = sqrt(mse)
        print('rmse', rmse)
        # store
        scores.append(rmse)
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
#     return score, scores
#     print('score, scores: ', score, scores)

    return score, scores


# In[15]:


# summarize scores
#Takes the name, model score and list of hourley mean scores
#print
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f]' % (name, score))


# In[16]:


# create a set of sarima configs to try
def sarima_configs():
    models = list()
    # define config lists
#     p_params = [0,1,2]
    p_params = [2]
    d_params = [0]
#     q_params = [0,1,2]        
    q_params = [24]
#     t_params = ['n','c','t','ct']
    t_params = ['c']
    P_params = [1]
    D_params = [0]
    Q_params = [1]
    m_params = [24]
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models


# In[17]:


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[18]:


# grid search configs
# Using train, test data and the list of configurations
# working in parallel


# def grid_search(data, cfg_list, n_test, parallel=True):
def grid_search(model_func, train, test, cfg_list, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing', verbose=1)
#         executor = Parallel(n_jobs=psutil.cpu_count(), verbose=1)
        tasks = (delayed(score_model)(model_func,train, test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
#         print('scores2', scores)
    else:
        scores = [score_model(model_func,train, test, cfg) for cfg in cfg_list]
#         print('scores1', scores)
    # remove empty results
#     print('scores', scores)
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# In[19]:


# score a model, return None on failure

def score_model(model_func,train, test, cfg, debug=False):
#     print('score_model')
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
#         result = walk_forward_validation(data, n_test, cfg)
        result = evaluate_model(model_func,train, test, cfg)[0]
#         print('DEBUG')
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = evaluate_model(model_func,train, test, cfg)[0]
#                 print('RESULT')
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# In[22]:


def train_channels_in_range_inclusive(a, b):
# Generating a dataframe for each channel
    best_config_dict = {}
#     for chan in chanlist[lower_limit:(upper_limit+1)]:
    for chan in chanlist[a:b+1]:
    #     # selecting only rows relating to the given channel
        d = hourly_data.loc[hourly_data.channel == chan]
    #    # removing partial days at start and end of sample
        df = full_days_only(d)
        # set size of out of sample test data
        oos_size = 24
        df, oos = out_of_sample(df, oos_size)
        # Generating train and test
        train, test, final_test = split_dataset(df)[0:3]
    
        # define the names and functions for the models we wish to evaluate
        models = dict()
        models['sarima'] = sarima_forecast

        print('channel', chan)
        # model configs
        n_test = 24
        cfg_list = sarima_configs()
        # print(cfg_list)
        # grid search
    #     count=0
        scores = grid_search(sarima_forecast, train, test, cfg_list)
#     print('channel: '+str(chan) +' done')
        # list top 3 configs
        for cfg, error in scores[:5]:
            print(cfg, error)
    #     print('SCORES',scores)    
        # best_config = scores[:1]
        best_config = ast.literal_eval(scores[:1][0][0])
        print('best config', best_config)
        ## For calculating out of sample score
#         best_oos_yhat = sarima_forecast(df, best_config)
#         oos_rmse = measure_rmse(final_test, best_oos_yhat)
#         print('Out of sample rmse: ', oos_rmse)
        # Add best config to the current best_config_dict
        best_config_dict[chan] = best_config
        print(best_config_dict)
    return best_config, best_config_dict


# In[23]:


# # Generating a dataframe for each channel
# for chan in chanlist[21:24]:
# #     # selecting only rows relating to the given channel
# #     d = data_grp_hourly.loc[data_grp_hourly.channel == chan]
#         # selecting only rows relating to the given channel
#     d = hourly_data.loc[hourly_data.channel == chan]
# #     print(d)
# #    # removing partial days at start and end of sample
# #     df = df.reset_index()
#     df = full_days_only(d)
# #     print('df', df)

#     # set size of out of sample test data
#     oos_size = 24
#     df, oos = out_of_sample(df, oos_size)
#     # Generating train and test
#     train, test, final_test = split_dataset(df)[0:3]


#     #Checking the size of the train and test sets
# #     print('train shape: ', train.shape)
# #     print('test shape: ', test.shape)
# #     print('final test shape',final_test.shape)
    
#     # define the names and functions for the models we wish to evaluate
#     models = dict()
#     models['sarima'] = sarima_forecast

#     # name each hour to be aggregated
# #     hours = ['0', '1', '2', '3', '4', '5', '6', '7', '8','9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
#     print('channel', chan)
#     # data split
#     n_test = 24
#     # model configs
#     cfg_list = sarima_configs()
# #     print(cfg_list)
#     # grid search
# #     count=0
#     scores = grid_search(sarima_forecast, train, test, cfg_list)
# #     print('channel: '+str(chan) +' done')
#     # list top 3 configs
#     for cfg, error in scores[:5]:
#         print(cfg, error)
#     # print('SCORES',scores)    
#     # best_config = scores[:1]
#     best_config = ast.literal_eval(scores[:1][0][0])
# #     print('best config', best_config)
# #     best_oos_yhat = sarima_forecast(df, best_config)
# #     oos_rmse = measure_rmse(final_test, best_oos_yhat)
# #     print('Out of sample rmse: ', oos_rmse)


# In[ ]:


train_channels_in_range_inclusive(0,2)


# In[92]:


best_config_dict


# In[ ]:




