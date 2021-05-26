import pandas as pd 
import numpy as np
from google.cloud import bigquery
import pickle
from datetime import datetime, timedelta
import sys


def get_location_data(path, columns):
    df = pd.read_csv(path, usecols=columns)
    return df

def get_loc(channel_id):
    '''
    Returns a dataframe consisting of geocoordinates and channel id of a device
    '''
    sql = """
    SELECT channel_id, longitude, latitude 
    FROM `airqo-250220.thingspeak.channel`
    WHERE channel_id={}
    """.format(channel_id)

    df = client.query(sql).to_dataframe()
    return df

def preprocessing(df): #hasn't yet been tested
    '''
    Preprocesses data for a particular channel
    '''
    df = df.sort_values(by='created_at',ascending=False)
    df = df.set_index('created_at')
    hourly_df = df.resample('H').mean()
    hourly_df.dropna(inplace=True)
    hourly_df= hourly_df.reset_index()
    return hourly_df

def get_entries_since(channel_id, start_date='2020-09-17 00:00:00', end_date='2020-09-17 02:00:00'):
    '''
    Returns data for a 7-day period for a particular channel
    '''
    client = bigquery.Client.from_service_account_json("C:/Users/User/AirQo-d982995f6dd8.json")

    sql = """
    SELECT created_at, channel_id, pm2_5 
    FROM `airqo-250220.thingspeak.clean_feeds_pms` 
    WHERE channel_id={} 
    AND created_at between '{}' AND '{}'
    """.format(channel_id, start_date, end_date)

    df = client.query(sql).to_dataframe() 
    return df  


if __name__=='__main__':
    locations_df = get_location_data('data/raw/channels.csv', ['location', 'id', 'lat', 'long'])
    channels = list(locations_df.id)
    print (channels)
    
