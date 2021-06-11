import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from theano import tensor as tt
import pymc3 as pm

from run_scripts.load_data import load_traintest_sparsereg

#Laplace prior PyMC3 model
def fit_mcmc_laplace(y,x,B,seed = 100,misspec = False):
    with pm.Model() as model:
        p = np.shape(x)[1]
        #Laplace
        b = pm.Gamma('b',alpha = 1,beta = 1)
        beta = pm.Laplace('beta',mu = 0, b = b,shape = p)
        intercept = pm.Flat('intercept')
        if misspec == True:
            sigma = pm.HalfNormal("sigma", sigma = 0.02) ## misspec prior
        else:
            sigma = pm.HalfNormal("sigma", sigma = 1) 
        obs = pm.Normal('obs',mu = pm.math.dot(x,beta)+ intercept,sigma = sigma,observed=y)

        trace = pm.sample(B,random_seed = seed, chains = 4)
    beta_post = trace['beta']
    intercept_post = trace['intercept'].reshape(-1,1)
    sigma_post = trace['sigma'].reshape(-1,1)
    b_post = trace['b'].reshape(-1,1)
    print(np.mean(sigma_post)) #check misspec.

    return beta_post,intercept_post,b_post,sigma_post


#Repeat 50 mcmc runs for different train test splits
def run_sparsereg_mcmc(dataset,misspec = False):
    #Repeat over 50 reps
    rep = 50
    train_frac = 0.7
    B = 2000

    #Initialize
    x,y,x_test,y_test,y_plot,n,d = load_traintest_sparsereg(train_frac,dataset,100)

    beta_post = np.zeros((rep,4*B, d))
    intercept_post = np.zeros((rep,4*B, 1))
    b_post =  np.zeros((rep,4*B, 1))
    sigma_post = np.zeros((rep,4*B,1))
    times = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100+j
        x,y,x_test,y_test,y_plot,n,d = load_traintest_sparsereg(train_frac,dataset,seed)
        start = time.time()
        beta_post[j],intercept_post[j],b_post[j],sigma_post[j] = fit_mcmc_laplace(y,x,B,seed,misspec)
        end = time.time()
        times[j] = (end - start)

    #Save posterior samples
    if misspec == False:
        suffix = dataset
    else:
        suffix = dataset + "_misspec"
    
    print("{}: {} ({})".format(suffix,np.mean(times), np.std(times)/np.sqrt(rep)))

    np.save("samples/beta_post_sparsereg_{}".format(suffix),beta_post)
    np.save("samples/intercept_post_sparsereg_{}".format(suffix),intercept_post)
    np.save("samples/b_post_sparsereg_{}".format(suffix),b_post)
    np.save("samples/sigma_post_sparsereg_{}".format(suffix),sigma_post)
    np.save("samples/times_sparsereg_{}".format(suffix),times)

