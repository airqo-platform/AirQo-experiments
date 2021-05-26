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
    df = pd.read_csv(path, usecols=columns)
    return df

def get_data(path):
    '''Loads data from saved pickle file'''
    d = pickle.load(open(path,'rb'))
    X = d['X']
    Y = d['Y']
    return X, Y

def train_model(Xtraining, Ytraining):
    '''
    Creates a model, trains it using given data and saves it for future use
    '''
    
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

def overall_function():
    rmse_list =[]

    #for i in range(5):
    for i in range(len(longs)):
        try:
            mean, var, Xtest, Ytest, rmse = cross_validation(X, Y, longs[i], lats[i])
            #print(f'Location {i} successful')
            #rmse_list.append(rmse)
            
            location_name = locations_df.loc[locations_df['long'] == longs[i], 'location'].iloc[0]
            print(f'{location_name} successful')
            rmse_list.append({'location':location_name, 'rmse':rmse})

            plt.figure(figsize=(12,6))
            plt.title(f'{location_name}, rmse:{rmse}')
            plt.xlim(f2(Xtest[:,2]).min()-timedelta(hours=1), f2(Xtest[:,2]).max()+timedelta(hours=1))
            plt.ylim(0,200)
            plt.plot(f2(Xtest[:, 2]), Ytest, label='Actual')
            plt.plot(f2(Xtest[:, 2]), mean, label='Predicted')
            plt.fill_between(f2(Xtest[:, 2]),
                            mean[:,0]-1.96*np.sqrt(var[:, 0]),
                            mean[:,0]+1.96*np.sqrt(var[:, 0]),
                            color="C0",
                            alpha=0.2)
            plt.legend(loc='best')
            plt.savefig(f'images/plots/{location_name}.png') 
            #plt.show()
        except Exception as e:
            print(f'Location {i} failed')
            print(e)
    return rmse_list

if __name__=='__main__':
    locations_df = get_location_data('data/raw/channels.csv', ['location', 'lat', 'long'])
    X, Y = get_data('data/raw/data.p')
    f = lambda time: Timestamp.fromtimestamp(time*3600)
    f2 = np.vectorize(f)
    longs = [X[:,0][index] for index in sorted(np.unique(X[:,0], return_index=True)[1])]
    lats = [X[:,1][index] for index in sorted(np.unique(X[:,1], return_index=True)[1])]
    rmse_list = overall_function()
    print(rmse_list)
