import pandas as pd 

def get_data(filepath):
    '''
    '''
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    '''
    '''
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
    df['hour_of_day'] = [datetime.fromisoformat(n).hour for n in df['created_at']]
    df['s2_pm2_5'] = df['s2_pm2_5'].mask(df['s2_pm2_5']==0).fillna(df['pm2_5'])
    df['s2_pm10'] = df['s2_pm10'].mask(df['s2_pm10']==0).fillna(df['pm10'])
    df.drop(columns=['site', 'created_at', 'landform_90m', 'dist_major_road'], axis=1, inplace=True)
    return df

def split_data(X, y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.45, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def train_model(model, Xtrain, ytrain):
    '''
    '''
    model.fit(Xtrain, ytrain)

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def predict(model, X, y):
    ypred = model.predict(X)
    rmse = round(mean_squared_error(y, ypred, squared=False), 2)
    return rmse, ypred