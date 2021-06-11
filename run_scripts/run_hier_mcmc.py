import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy as sp
from theano import tensor as tt
import pymc3 as pm

from run_scripts.load_data import gen_data_hier,load_traintest_hier

#Hierarchical PyMC3 model
def fit_mcmc_hier(y,x,K,B,seed):
    with pm.Model() as model:
        #Hyperpriors:
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        sigma_b = pm.Exponential("sigma_b", 1.0)

        #Varying intercepts:
        za_group = pm.Normal("za_group", mu=0.0, sigma=1.0, shape = K)
        #Varying slopes:
        zb_group = pm.Normal("zb_group", mu=0.0, sigma=1.0, shape = K)

        #Mean:
        x_ = pm.Data("x_", x[:,0])
        group_index = pm.Data("group_index", x[:,1].astype('int'))
        theta = (a + za_group[tt.cast(group_index,'int8')] * sigma_a) + (b + zb_group[tt.cast(group_index,'int8')] * sigma_b) * x_

        #Likelihood:
        sigma = pm.Exponential("sigma", 1.)
        obs = pm.Normal("obs", mu = theta, sigma=sigma, observed=y)
        trace = pm.sample(B,random_seed=seed,tune=2000, target_accept=0.99, chains = 4)

    #Reparametrize to parameters of interest
    beta_post = np.array(trace['zb_group']*trace['sigma_b'].reshape(-1,1) + trace['b'].reshape(-1,1))
    intercept_post = np.array(trace['za_group']*trace['sigma_a'].reshape(-1,1)+trace['a'].reshape(-1,1))
    a_post = np.array(trace['a'])
    sigma_a_post = np.array(trace['sigma_a'])
    b_post = np.array(trace['b'])
    sigma_b_post = np.array(trace['sigma_b'])
    sigma_post = np.array(trace['sigma']).reshape(-1,1)

    return beta_post,intercept_post, sigma_post

#Repeat 50 mcmc runs for different train test splits
def run_hier_mcmc(dataset,misspec= False):
    #Repeat over 50 reps
    rep = 50
    B = 2000

    #Initialize
    if dataset == 'sim':
        seed = 100
        K = 5
        p = 1
        n = 10 
        n_test = 10 

        y,x,y_test,x_test,beta_true,sigma_true,y_plot = gen_data_hier(n,p,n_test,seed,K, misspec = misspec)

    elif dataset =='radon':
        train_frac = 1.0
        rep =1
        x,y,x_test,y_test,y_plot,n,d = load_traintest_hier(1.0,dataset,100)
        K = np.shape(np.unique(x[:,1]))[0]

    beta_post = np.zeros((rep,4*B,K))
    intercept_post = np.zeros((rep,4*B, K))
    sigma_post = np.zeros((rep,4*B,1))
    times = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100+j
        if dataset =='sim':
            y,x,y_test,x_test,beta_true,sigma_true,y_plot = gen_data_hier(n,p,n_test,seed,K, misspec = misspec)
        elif dataset =='radon':
            x,y,x_test,y_test,y_plot,n,d = load_traintest_hier(train_frac,dataset,seed)

        start = time.time()
        beta_post[j],intercept_post[j],sigma_post[j] = fit_mcmc_hier(y,x,K,B,seed)
        print(np.mean(sigma_post[j]))
        end = time.time()
        times[j] = end- start

    #Save posterior samples
    #Load posterior samples
    if misspec == False:
        suffix = dataset
    else:
        suffix = dataset + "_misspec"

    np.save("samples/beta_post_hier_{}".format(suffix),beta_post)
    np.save("samples/intercept_post_hier_{}".format(suffix),intercept_post)
    np.save("samples/sigma_post_hier_{}".format(suffix),sigma_post)
    np.save("samples/times_hier_{}".format(suffix),times)

    print("{}: {} ({})".format(suffix, np.mean(times), np.std(times)/np.sqrt(rep)))


