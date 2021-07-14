import pandas as pd 



def cross_validation(X, Y, long, lat):
    
    location_indices = np.where(np.logical_and(X[:,0]==long, X[:,1]==lat))
    
    Xtraining = X[np.logical_not(np.logical_and(X[:,0]==long, X[:,1]==lat))]
    Ytraining = np.delete(Y, slice(location_indices[0][0],location_indices[0][-1]+1), axis=0)
    
    Xtest = X[np.logical_and(X[:,0]==long, X[:,1]==lat)]
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