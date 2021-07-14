import numpy as np
import gpflow
from gpflow import set_trainable

def preprocessing(df, lat, lon, start_date, end_date):
    df.set_index('time', inplace=True)
    df.dropna(inplace=True)
    hourly_df = df.resample('H').mean()
    hourly_df['latitude'], hourly_df['longitude']=lat,lon
    hourly_df.reset_index(inplace=True)
    hourly_df.dropna(inplace=True)
    hourly_df = hourly_df[(hourly_df['time']>= start_date)&(hourly_df['time']<=end_date)]
    hourly_df['day'] = [time.day for time in hourly_df['time']]
    hourly_df['day_of_week'] = [time.weekday() for time in hourly_df['time']]
    hourly_df['hour'] = [time.hour for time in hourly_df['time']]
    #hourly_df['time'] = [time.timestamp()/3600 for time in hourly_df['time']]
    hourly_df.drop(['time'], axis=1, inplace=True)
    return hourly_df

def cross_validation(X, Y, lon, lat):
    
    location_indices = np.where(np.logical_and(X[:,0]==lon, X[:,1]==lat))
    
    Xtraining = X[np.logical_not(np.logical_and(X[:,0]==lon, X[:,1]==lat))]
    Ytraining = np.delete(Y, slice(location_indices[0][0],location_indices[0][-1]+1), axis=0)
    
    Xtest = X[np.logical_and(X[:,0]==lon, X[:,1]==lat)]
    Ytest = Y[location_indices[0][0]:location_indices[0][-1]+1]
    #kernel
    k = gpflow.kernels.RBF(variance=625) + gpflow.kernels.Bias()
    #training model
    m = gpflow.models.GPR(data=(Xtraining, Ytraining), kernel=k, mean_function=None)
    m.likelihood.variance.assign(400)
    set_trainable(m.kernel.kernels[0].variance, False)
    set_trainable(m.likelihood.variance, False)
    #optimization
    opt = gpflow.optimizers.Scipy()
    def objective_closure():
        return - m.log_marginal_likelihood()
    
    opt_logs = opt.minimize(objective_closure,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    mean, var = m.predict_f(Xtest)
        
    rmse = sqrt(mean_squared_error(Ytest, mean.numpy()))
    
    return mean.numpy(), var.numpy(), Xtest, Ytest, round(rmse, 2)