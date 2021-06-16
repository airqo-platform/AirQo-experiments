#importing packages
from pymongo import MongoClient
from datetime import datetime, timedelta
import requests


client = MongoClient("mongodb://admin:airqo-250220-master@35.224.67.244:27017")

def connect_db(owner):
    db_name = f'airqo_netmanager_{owner}'
    db = client[db_name]
    return db

def get_device_details(device_id, owner):
    '''
    Returns a device\'s details given channel ID
    '''
    db= connect_db(owner)    
    query = {
        'channelID': device_id
    }
    projection = {
        '_id': 0,
        'latitude': 1,
        'longitude': 1,
        'name':1,
        'channelID':1
    }
    records = list(db.devices.find(query, projection))
    return records[0]['latitude'], records[0]['longitude'], records[0]['name']

def str_to_date(st):
    return datetime.strptime(st,'%Y-%m-%dT%H:%M:%S.%fZ')


def date_to_str(mydate):
    return datetime.strftime(mydate,'%Y-%m-%dT%H:%M:%SZ')


def get_pm_data(device_id, owner,start_time='2021-01-01T01:00:00Z',end_time=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')):
    '''
    Gets the PM data of a particular device in a specified time period
    '''
    lat, lon, name = get_device_details(device_id, owner)
    url = 'https://staging-platform.airqo.net/api/v1/devices/events'
    result = [] #stores all data downloaded for a device
    measurements_length= 120
    count = 0
    while measurements_length==120:
        count+=1
        parameters = {
            'tenant': owner,
            'device': name,
            'startTime': start_time,
            'endTime': end_time,
            'recent': 'no'
        }
        print(f'Iteration {count} - Start Time: {start_time}, End Time: {end_time}')
        try:
            response = requests.get(url, params=parameters)
            if response.status_code ==200:
                response_json = response.json()
                measurements = response_json['measurements']
                measurements_length = len(measurements)
                if measurements_length!=0:
                    result.extend(measurements)
                    new_end_time = measurements[-1]['time']
                    end_time = date_to_str(str_to_date(new_end_time) - timedelta(seconds=1))
        except Exception as e:
            #print(e)
            pass
    modified_result = [{'time': x['time'],
                     'latitude': lat,
                     'longitude': lon,
                     'pm2_5': x['pm2_5']['value'],
                     #'calibrated_pm2_5': x['pm2_5']['calibratedValue'],
                     'pm10': x['pm10']['value'],
                    } for x in result]
    return modified_result
        