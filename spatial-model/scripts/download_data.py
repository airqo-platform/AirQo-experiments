import pandas as pd 
import numpy as np
from google.cloud import bigquery
import pickle
from datetime import datetime, timedelta
import sys


def get_location_data(path, columns):
    """Gets details of locations considered

    Parameters
    ----------
    path : str
        The file location of the spreadsheet
    columns: list
        The names of the columns to be considered

    Returns
    -------
    df : DataFrame
        a pandas dataframe consisting of the locations' data in the spreadsheet
    """
    df = pd.read_csv(path, usecols=columns)
    return df

def get_loc(channel_id):
    """Gets details of locations considered

    Parameters
    ----------
    channel_id : str
        The channel ID of the device

    Returns
    -------
    df : DataFrame
        a pandas dataframe consisting of the device's geographical coordinates and channel ID
    """

    sql = """
    SELECT channel_id, longitude, latitude 
    FROM `airqo-250220.thingspeak.channel`
    WHERE channel_id={}
    """.format(channel_id)

    df = client.query(sql).to_dataframe()
    return df

def preprocessing(df):
    """Preprocesses data for a particular device

    Parameters
    ----------
    df : DataFrame
        a panda dataframe that has a partuclar device's data

    Returns
    -------
    df : DataFrame
        a pandas dataframe with the preprocessed data of the device
    """
    df = df.sort_values(by='created_at',ascending=False)
    df = df.set_index('created_at')
    hourly_df = df.resample('H').mean()
    hourly_df.dropna(inplace=True)
    hourly_df= hourly_df.reset_index()
    return hourly_df

def get_entries_since(channel_id, start_date='2020-09-17 00:00:00', end_date='2020-09-23 23:59:59'):
    """Gets a device's data between a specified period

    Parameters
    ----------
    channel_id : str
        The channel ID of a device
    start_date: str
        The start date of the peiod of interest
    end_date:str
        The end date of the period of interest

    Returns
    -------
    df : DataFrame
        a pandas dataframe consisting of a device's data within a specified period
    """
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
    df = get_entries_since(channels[5])
    print (df.head())
    
