import time
import numpy as np
import scipy as sp
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
import pandas as pd

#import from cb package
from run_scripts.load_data import gen_data_hier,load_traintest_hier
from conformal_bayes import conformal_Bayes_functions as cb
from conformal_bayes import Bayes_MCMC_functions as bmcmc

#Conformalized Bayes for grouped data
def run_hier_conformal(dataset,misspec):
    #Compute intervals
    #Initialize
    if dataset == 'sim':
        seed = 100
        K = 5
        p = 1
        n = 10 
        n_test_pergrp = 10 
        rep = 50
        B = 4*2000
        y,x,y_test,x_test,beta_true,sigma_true,y_plot = gen_data_hier(n,p,n_test_pergrp,seed,K,misspec = misspec)

    elif dataset =='radon':
        train_frac = 1.
        x,y,x_test,y_test,y_plot,n,d = load_traintest_hier(1,dataset,100)
        K = np.shape(np.unique(x[:,1]))[0]
        rep = 1
        B = 4*2000
        x,y,x_test,y_test,y_plot,n,d = load_traintest_hier(train_frac,dataset,100)
        #Load all possible x_test and group assignments
        x_test = np.zeros((2*K,2))
        for k in range(K):
            x_test[2*k:2*k + 2,1] = k
            x_test[2*k,0] = 0
            x_test[2*k+1,0]= 1
        n_test = np.shape(x_test)[0]
        y_test = np.zeros(n_test) #Place holder

    #Compute intervals
    alpha = 0.2
    dy = y_plot[1]- y_plot[0]

    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep,n_test))
    coverage_cb_exact = np.zeros((rep,n_test))
    coverage_bayes = np.zeros((rep,n_test))

    length_cb = np.zeros((rep,n_test))
    length_bayes = np.zeros((rep,n_test))
        
    band_bayes = np.zeros((rep,n_test,2))
    region_cb = np.zeros((rep,n_test,np.shape(y_plot)[0]))  

    times_bayes = np.zeros(rep)
    times_cb = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100+j

        #Load data
        if dataset =='sim':
            y,x,y_test,x_test,beta_true,sigma_true,y_plot = gen_data_hier(n,p,n_test_pergrp,seed,K,misspec = misspec)
        elif dataset =='radon':
            x,y,x_test,y_test,y_plot,n,d = load_traintest_hier(train_frac,dataset,seed)
            x_test = np.zeros((2*K,2))
            #Load all possible x_test and group assignments
            for k in range(K):
                x_test[2*k:2*k + 2,1] = k
                x_test[2*k,0] = 0
                x_test[2*k+1,0]= 1
            n_test = np.shape(x_test)[0]
            y_test = np.zeros(n_test) #Place holder

        #Precompute log likelihood terms
        #Load posterior samples (for each j to save memory)
        if misspec == False:
            suffix = dataset
        else:
            suffix = dataset + "_misspec"

        beta_post = jnp.array(np.load("samples/beta_post_hier_{}.npy".format(suffix)))[j]
        intercept_post = jnp.array(np.load("samples/intercept_post_hier_{}.npy".format(suffix)))[j]
        sigma_post = jnp.array(np.load("samples/sigma_post_hier_{}.npy".format(suffix)))[j]

        #Bayes
        start = time.time()
        @jit
        def normal_likelihood_cdf(y,x):
            group = x[:,-1].astype('int32') 
            x_0 = x[:,0] 
            return norm.cdf(y,loc = beta_post[:,group] * x_0.transpose()+ intercept_post[:,group]\
                            ,scale = sigma_post) #compute likelihood samples

        #Precompute cdfs
        cdf_test =  normal_likelihood_cdf(y_plot.reshape(-1,1,1),x_test)

        for i in (range(n_test)):
            group_ind = x_test[i,1].astype('int')
            
            #Compute Bayes interval
            band_bayes[j,i] = bmcmc.compute_bayes_band_MCMC(alpha,y_plot,cdf_test[:,:,i])
            coverage_bayes[j,i] = (y_test[i] >=band_bayes[j,i,0])&(y_test[i] <=band_bayes[j,i,1])
            length_bayes[j,i] = np.abs(band_bayes[j,i,1]- band_bayes[j,i,0])   
        end = time.time()
        times_bayes[j] =  (end- start)


        #Conformal Bayes
        start = time.time()
        #Define likelihood from posterior samples
        @jit
        def normal_loglikelihood(y,x):
            group = x[:,-1].astype('int32') 
            x_0 = x[:,0] 
            return norm.logpdf(y,loc = beta_post[:,group]* x_0.transpose() + intercept_post[:,group]\
                               ,scale = sigma_post) #compute likelihood samples

        #Compute loglikelihood across groups
        groups_train = np.unique(x[:,1]).astype('int')
        K_train = np.size(groups_train)
        n_groups = np.zeros(K_train)
        logp_samp_n = []

        for k in (range(K_train)):
            ind_group = (x[:,1] == groups_train[k])
            n_groups[k] =np.sum(ind_group)
            logp_samp_n.append(normal_loglikelihood(y[ind_group],x[ind_group]))

        logwjk = normal_loglikelihood(y_plot.reshape(-1,1,1),x_test)
        logwjk_test = normal_loglikelihood(y_test,x_test).reshape(1,-1,n_test)

        #Compute conformal regions
        for i in (range(n_test)):
            group_ind = x_test[i,1].astype('int')
                    
            #old group
            if group_ind in groups_train: 
                 #within group nonconform measure
                region_cb[j,i] = cb.compute_cb_region_IS(alpha,logp_samp_n[np.where(group_ind == groups_train)[0][0]],logwjk[:,:,i])

            else: #new group
                print('Error: test group not in training')
                return

            coverage_cb[j,i] = region_cb[j,i,np.argmin(np.abs(y_test[i]-y_plot))]
            length_cb[j,i] = np.sum(region_cb[j,i])*dy
        end = time.time()
        times_cb[j] =  (end- start)

        #Compute exact coverage
        for i in (range(n_test)):
            group_ind = x_test[i,1].astype('int')
                    
            #old group
            if group_ind in groups_train: 
                #within group nonconform measure
                coverage_cb_exact[j,i] = cb.compute_cb_region_IS(alpha,logp_samp_n[np.where(group_ind == groups_train)[0][0]],logwjk_test[:,:,i])
            else: #new group
                print('Error: test group not in training')
                return

    ## Save per group results ##
    #Find indices for each group
    coverage_cb_grp = np.zeros((K_train, rep))
    coverage_cb_exact_grp = np.zeros((K_train, rep))
    coverage_bayes_grp =  np.zeros((K_train, rep))
    length_cb_grp =  np.zeros((K_train, rep))
    length_bayes_grp = np.zeros((K_train, rep))

    for j in tqdm(range(rep)):
        #Load data
        if dataset =='sim':
            y,x,y_test,x_test,beta_true,sigma_true,y_plot = gen_data_hier(n,p,n_test_pergrp,seed,K,misspec = misspec)
        elif dataset =='radon':
            x,y,x_test,y_test,y_plot,n,d = load_traintest_hier(train_frac,dataset,seed)
            x_test = np.zeros((2*K,2))
            #Load all possible x_test and group assignments
            for k in range(K):
                x_test[2*k:2*k + 2,1] = k
                x_test[2*k,0] = 0
                x_test[2*k+1,0]= 1
            n_test = np.shape(x_test)[0]
            y_test = np.zeros(n_test) #Place holder

        for k in (range(K_train)):
            coverage_cb_grp[k, j] = np.mean(coverage_cb[j,x_test[:,-1] ==k])
            coverage_cb_exact_grp[k, j] = np.mean(coverage_cb_exact[j,x_test[:,-1] ==k])
            coverage_bayes_grp[k, j] = np.mean(coverage_bayes[j,x_test[:,-1] ==k])
            length_cb_grp[k, j] = np.mean(length_cb[j,x_test[:,-1] ==k])
            length_bayes_grp[k, j] = np.mean(length_bayes[j,x_test[:,-1] ==k])

    #Save regions
    np.save("results/region_cb_hier_{}".format(suffix),region_cb)
    np.save("results/band_bayes_hier_{}".format(suffix),band_bayes)

    np.save("results/coverage_cb_hier_{}".format(suffix),coverage_cb)
    np.save("results/coverage_cb_exact_hier_{}".format(suffix),coverage_cb_exact)
    np.save("results/coverage_bayes_hier_{}".format(suffix),coverage_bayes)

    np.save("results/length_cb_hier_{}".format(suffix),length_cb)
    np.save("results/length_bayes_hier_{}".format(suffix),length_bayes)

    np.save("results/coverage_cb_grp_hier_{}".format(suffix),coverage_cb_grp)
    np.save("results/coverage_cb_exact_grp_hier_{}".format(suffix),coverage_cb_exact_grp)
    np.save("results/coverage_bayes_grp_hier_{}".format(suffix),coverage_bayes_grp)

    np.save("results/length_cb_grp_hier_{}".format(suffix),length_cb_grp)
    np.save("results/length_bayes_grp_hier_{}".format(suffix),length_bayes_grp)

    np.save("results/times_cb_hier_{}".format(suffix),times_cb)
    np.save("results/times_bayes_hier_{}".format(suffix),times_bayes)

