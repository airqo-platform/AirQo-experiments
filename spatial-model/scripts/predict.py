import pickle
import gpflow
import numpy as np
from gpflow import set_trainable
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Timestamp
from datetime import timedelta
from statistics import mean

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

def get_data(path):
    """Loads data from saved file

    Parameters
    ----------
    path : str
        The file location of the pickle file

    Returns
    -------
    X : Array
        a numpy array consisting of the features data
    Y: Array
        a numpy array consisting of the target data
    """
    d = pickle.load(open(path,'rb'))
    X = d['X']
    Y = d['Y']
    return X, Y

def train_model(Xtraining, Ytraining):
    """Develops and trains a model

    Parameters
    ----------
    Xtraining : Array
        The X training data
    Ytraining: Array
        The target training data

    Returns
    -------
    m : model
        a GPR model trained on the training data
    """
    
    #defining kernel
    k = gpflow.kernels.RBF(lengthscales=[0.08, 0.08, 1.], variance=625) + gpflow.kernels.Bias()
    #defining model
    m = gpflow.models.GPR(data=(Xtraining, Ytraining), kernel=k, mean_function=None)
    m.likelihood.variance.assign(400)
    set_trainable(m.kernel.kernels[0].lengthscales, False)
    set_trainable(m.kernel.kernels[0].variance, False)
    set_trainable(m.likelihood.variance, False)
    #optimization
    opt = gpflow.optimizers.Scipy()
    def objective_closure():
             return - m.log_marginal_likelihood()

    opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
    return m

def make_predictions(m, Xtest, Ytest):
    mean, var = m.predict_f(Xtest)
    rmse = sqrt(mean_squared_error(Ytest, mean.numpy()))
    return mean, var, rmse

def cross_validation(X, Y, long, lat):
    location_indices = np.where(np.logical_and(X[:,0]==long, X[:,1]==lat))
    
    Xset = X[np.logical_not(np.logical_and(X[:,0]==long, X[:,1]==lat))]
    Yset = np.delete(Y, slice(location_indices[0][0],location_indices[0][-1]+1), axis=0)
        
    Xtraining = Xset[::2,:]
    Ytraining = Yset[::2,:]
    
    Xtest = X[np.logical_and(X[:,0]==long, X[:,1]==lat)]
    Ytest = Y[location_indices[0][0]:location_indices[0][-1]+1]

    m = train_model(Xtraining, Ytraining)
    mean, var, rmse = make_predictions(m, Xtest, Ytest) 
    return mean.numpy(), var.numpy(), Xtest, Ytest, round(rmse, 2)


if __name__=='__main__':
    locations_df = get_location_data('data/raw/channels.csv', ['location', 'lat', 'long'])
    X, Y = get_data('data/raw/data.p')
    print(locations_df.shape[0], X.shape, Y.shape)
