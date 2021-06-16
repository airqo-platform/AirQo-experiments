# Code used to download data (if applicable)
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

