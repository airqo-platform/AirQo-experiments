import pandas as pd 
import numpy as np
from google.cloud import storage
import pickle
import json
import requests
from datetime import datetime, timedelta

def get_channels():
    '''
    Gets details of channels whose data is to be used in training
    '''
    blob = storage_client.get_bucket('api-keys-bucket') \
        .get_blob('kampala-api-keys.json') \
        .download_as_string()
    return json.loads(blob)

def preprocessing(df):
    '''
    Preprocesses data for a particular channel
    '''
    df = df.drop_duplicates()
    df['field1'].fillna(df['field3'], inplace=True)
    df = df[['created_at','field1']]
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['field1'] = pd.to_numeric(df['field1'], errors='coerce')
    df = df.sort_values(by='created_at',ascending=False)
    df = df.set_index('created_at')
    hourly_df = df.resample('H').mean()
    hourly_df.dropna(inplace=True)
    hourly_df= hourly_df.reset_index()

    return hourly_df

def download_seven_days(channel_id, api_key):
    '''
    Downloads data from ThingSpeak for a specific channel for the past week
    '''
    channel_data = []
    result = 8000
    end_time = datetime.utcnow()
    start_time = end_time-timedelta(days=7)
    #start_time = end_time-timedelta(hours=1) #delete
    start = datetime.strftime(start_time, '%Y-%m-%dT%H:%M:%SZ')
    end = datetime.strftime(end_time, '%Y-%m-%dT%H:%M:%SZ')
    base_url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json'
    
    while (end_time > start_time) and (result==8000):
        channel_url = base_url+'?start='+start+'&end='+end
        print(channel_url)
        data = json.loads(requests.get(channel_url, timeout = 100.0).content.decode('utf-8'))
        if (data!=-1 and len(data['feeds'])>0):
            channel_data.extend(data['feeds'])
            end = data['feeds'][0]['created_at']
            result = len(data['feeds'])
            print(result)
            end_time = datetime.strptime(end, '%Y-%m-%dT%H:%M:%SZ') - timedelta(seconds=1)
        else:
            return pd.DataFrame()
    return pd.DataFrame(channel_data)

def download_all_data():
    '''
    Re-trains the model regularly
    '''
    X = np.zeros([0,3])
    Y = np.zeros([0,1])
    channels = get_channels()
    for channel in channels:
        d = download_seven_days(channel['id'], channel['api_key'])
        if d.shape[0]!=0:
            d = preprocessing(d)
            df = pd.DataFrame({'channel_id':[channel['id']], 
                               'longitude':[channel['long']], 
                               'latitude':[channel['lat']]})
        
            Xchan = np.c_[np.repeat(np.array(df)[:,1:],d.shape[0],0),[n.timestamp()/3600 for n in d['created_at']]]
            Ychan = np.array(d['field1'])
            X = np.r_[X,Xchan]
            Y = np.r_[Y,Ychan[:, None]]
    pickle.dump({'X':X,'Y':Y},open('data/raw/data.p','wb'))


if __name__=='__main__':
    storage_client = storage.Client.from_service_account_json('AirQo-d982995f6dd8.json')
    download_all_data()