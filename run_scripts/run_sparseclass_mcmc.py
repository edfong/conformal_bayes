import time
import numpy as np
import pandas as pd
from theano import tensor as tt
import pymc3 as pm
from tqdm import tqdm

from run_scripts.load_data import load_traintest_sparseclass

#Laplace prior PyMC3 model
def fit_mcmc_laplace(y,x,B,seed = 100):
    with pm.Model() as model:
        p = np.shape(x)[1]
        #Laplace
        b = pm.Gamma('b',alpha = 1,beta = 1)
        beta = pm.Laplace('beta',mu = 0, b = b,shape = p)
        intercept = pm.Flat('intercept')
        obs = pm.Bernoulli('obs',logit_p = pm.math.dot(x,beta)+ intercept,observed=y)
        trace = pm.sample(B,random_seed = seed,chains = 4)
        
    beta_post = trace['beta']
    intercept_post = trace['intercept'].reshape(-1,1)
    b_post = trace['b'].reshape(-1,1)

    return beta_post,intercept_post,b_post

#Repeat 50 mcmc runs for different train test splits
def run_sparseclass_mcmc(dataset):
    #Repeat over 50 reps
    rep = 50
    train_frac = 0.7
    B = 2000

    #Initialize
    x,y,x_test,y_test,y_plot,n,d = load_traintest_sparseclass(train_frac,dataset,100)

    beta_post = np.zeros((rep,4*B, d))
    intercept_post = np.zeros((rep,4*B, 1))
    b_post =  np.zeros((rep,4*B, 1))
    times = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100+j
        x,y,x_test,y_test,y_plot,n,d = load_traintest_sparseclass(train_frac,dataset,seed)
        start = time.time()
        beta_post[j],intercept_post[j],b_post[j] = fit_mcmc_laplace(y,x,B,seed)
        end = time.time()
        times[j] = (end - start)

    print("{}: {} ({})".format(dataset, np.mean(times), np.std(times)/np.sqrt(rep)))

    #Save posterior samples
    np.save("samples/beta_post_sparseclass_{}".format(dataset),beta_post)
    np.save("samples/intercept_post_sparseclass_{}".format(dataset),intercept_post)
    np.save("samples/b_post_sparseclass_{}".format(dataset),b_post)
    np.save("samples/times_sparseclass_{}".format(dataset),times)
