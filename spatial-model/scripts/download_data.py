import pandas as pd 
import numpy as np
from google.cloud import bigquery
import pickle
from datetime import datetime, timedelta
import predict

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
    from datetime import datetime,timedelta

    sql = """
    SELECT created_at, channel_id, pm2_5 
    FROM `airqo-250220.thingspeak.clean_feeds_pms` 
    WHERE channel_id={} 
    AND created_at between '{}' AND '{}'
    """.format(channel_id, start_date, end_date)

    df = client.query(sql).to_dataframe() 
    return df

def download_all_data():
    #Arrays to hold X and Y data
    X = np.zeros([0,3])
    Y = np.zeros([0,1])

    #get data for each channel and append it to X and Y arrays respectively
    for chan in channels:
        d = get_entries_since(chan)
        if d.shape[0]!=0:
            d = preprocessing(d)
            #loc = get_loc(chan)
            loc = locations_df.loc[locations_df['id'] == chan, ['id', 'lat', 'long']]
            loc = loc[['id', 'long', 'lat']]
                
            Xchan = np.c_[np.repeat(np.array(loc)[:,1:],d.shape[0],0),[n.timestamp()/3600 for n in d['created_at']]]
                
            Ychan = np.array(d['pm2_5'])
            X = np.r_[X,Xchan]#appending device X data to array X
            Y = np.r_[Y,Ychan[:, None]]#appending device Y data to array Y
            print (str(chan)+':done!')
        else:
            print(str(chan)+':empty!')

    pickle.dump({'X':X,'Y':Y},open('data/raw/data.p','wb'))


if __name__=='__main__':
    #storage_client = storage.Client.from_service_account_json('AirQo-d982995f6dd8.json')
    #download_all_data()
    locations_df = predict.get_location_data('data/raw/channels.csv', ['location', 'id', 'lat', 'long'])
    channels = list(locations_df.id)
    client = bigquery.Client.from_service_account_json("C:/Users/User/AirQo-d982995f6dd8.json")
    download_all_data()