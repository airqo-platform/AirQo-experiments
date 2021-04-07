import pandas as pd 

def get_data(filepath):
    """Creates a dataframe from a spreadsheet

    Parameters
    ----------
    filepath : str
        The file location of the spreadsheet

    Returns
    -------
    df : DataFrame
        a pandas dataframe consisting of the data in the spreadsheet
    """
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Applies some preprocessing steps to existing dataframe

    Parameters
    ----------
    df : DataFrame
        TA pandas dataframe containing raw data

    Returns
    -------
    df : DataFrame
        a pandas dataframe containing preprocessed/clean data
    """
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
    df['hour_of_day'] = [datetime.fromisoformat(n).hour for n in df['created_at']]
    df['s2_pm2_5'] = df['s2_pm2_5'].mask(df['s2_pm2_5']==0).fillna(df['pm2_5'])
    df['s2_pm10'] = df['s2_pm10'].mask(df['s2_pm10']==0).fillna(df['pm10'])
    df.drop(columns=['site', 'created_at', 'landform_90m', 'dist_major_road'], axis=1, inplace=True)
    return df

def split_data(X, y):
    """Splits given data into training and test sets

    Parameters
    ----------
    X : DataFrame
        The features columns of the data
    y : Series
        The target column of the data

    Returns
    -------
    Xtrain : DataFrame
        the X training data
    Xtest : DataFrame
        the X test data
    ytrain : Series
        the y training data
    ytest : Series
        the y test data
    """
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.45, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def train_model(model, Xtrain, ytrain):
    """Fits the model onto the training data

    Parameters
    ----------
    model: ML model
    Xtrain : DataFrame
        Feature columns training data
    ytrain : Series
        Target column training data

    """
    
    model.fit(Xtrain, ytrain)

def save_model(model, filepath):
    """Saves the model onto disk space

    Parameters
    ----------
    model: ML model
    filepath : str
        File location where model is to be stored
    
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """Loads an already saved model

    Parameters
    ----------
    filepath : str
        File location where model is stored
    
    Returns
    -------
    model : ML model
        the saved ML model
    
    """
    model = joblib.load(filepath)
    return model

def predict(model, X, y):
    """Loads an already saved model

    Parameters
    ----------
    model : str
        File location where model is stored
    X : DataFrame
        The feature columns test data
    Y: Series
        The target column test data

    Returns
    -------
    rmse : float
        the root mean squared error of the model
    ypred: Series
        the model's predicted target values    
    """
    ypred = model.predict(X)
    rmse = round(mean_squared_error(y, ypred, squared=False), 2)
    return rmse, ypred