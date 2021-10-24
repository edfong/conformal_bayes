import time
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
from sklearn.linear_model import LassoCV,Lasso
import pandas as pd

#import from cb package
from run_scripts.load_data import load_traintest_sparsereg
from conformal_bayes import conformal_Bayes_functions as cb
from conformal_bayes import Bayes_MCMC_functions as bmcmc

#Define baselines
#Lasso split method
def conformal_split(y,x,x_test,alpha,y_plot,seed=100):
    n = np.shape(y)[0]
    n_test = np.shape(x_test)[0]
    #Fit lasso to training set
    ls = LassoCV(cv = 5,random_state = seed)
    n_train = int(n/2)
    ls.fit(x[0:n_train],y[0:n_train])
    #Predict lasso on validation set
    y_pred_val = ls.predict(x[n_train:])
    resid = np.abs(y_pred_val - y[n_train:])
    k = int(np.ceil((n/2 + 1)*(1-alpha)))
    d = np.sort(resid)[k-1]
    #Compute split conformal interval
    band_split = np.zeros((n_test,2))
    y_pred_test = ls.predict(x_test) #predict lasso on test
    band_split[:,0] = y_pred_test - d
    band_split[:,1] = y_pred_test + d
    return band_split
    
#Lasso full method
def conformal_full(y,x,x_test,alpha,y_plot,C,seed=100):
    n = np.shape(y)[0]
    rank_full = np.zeros(np.shape(y_plot)[0])
    for i in range(np.shape(y_plot)[0]):
        y_new = y_plot[i]
        x_aug = np.concatenate((x,x_test),axis = 0)
        y_aug = np.append(y,y_new)
        ls = Lasso(alpha = C,random_state = seed)
        ls.fit(x_aug,y_aug)
        y_pred_val = ls.predict(x_aug)
        resid = np.abs(y_pred_val - y_aug)
        rank_full[i] = np.sum(resid>=resid[-1])/(n+1)
    region_full = rank_full > alpha 
    return region_full

#Main run function for sparse regression
def run_sparsereg_conformal(dataset,misspec = False):
    #Compute intervals
    #Initialize
    train_frac = 0.7
    x,y,x_test,y_test,y_plot,n,d = load_traintest_sparsereg(train_frac,dataset,100)

    #Load posterior samples
    if misspec == False:
        suffix = dataset
    else:
        suffix = dataset + "_misspec"

    beta_post = jnp.load("samples/beta_post_sparsereg_{}.npy".format(suffix))
    intercept_post = jnp.load("samples/intercept_post_sparsereg_{}.npy".format(suffix))
    sigma_post = jnp.load("samples/sigma_post_sparsereg_{}.npy".format(suffix))

    #Initialize
    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep,n_test))
    coverage_cb_exact = np.zeros((rep,n_test)) #avoiding grid effects
    coverage_bayes = np.zeros((rep,n_test))
    coverage_split = np.zeros((rep,n_test))
    coverage_full = np.zeros((rep,n_test))

    length_cb = np.zeros((rep,n_test))
    length_bayes = np.zeros((rep,n_test))
    length_split = np.zeros((rep,n_test))
    length_full = np.zeros((rep,n_test))
        
    band_bayes = np.zeros((rep,n_test,2))
    region_cb = np.zeros((rep,n_test,np.shape(y_plot)[0]))
    region_full = np.zeros((rep,n_test,np.shape(y_plot)[0]))
    band_split = np.zeros((rep,n_test,2))

    times_bayes = np.zeros(rep)
    times_cb = np.zeros(rep)
    times_split = np.zeros(rep)
    times_full = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j
        #load dataset
        x,y,x_test,y_test,y_plot,n,d = load_traintest_sparsereg(train_frac,dataset,seed)
        dy = y_plot[1] - y_plot[0]

        #split method
        start = time.time()
        band_split[j] = conformal_split(y,x,x_test,alpha,y_plot,seed)
        coverage_split[j] = (y_test >=band_split[j,:,0])&(y_test <=band_split[j,:,1])
        length_split[j] = np.abs(band_split[j,:,0] - band_split[j,:,1])
        end = time.time()
        times_split[j]= end - start

        #full method
        start = time.time()
        C = 0.004
        for i in (range(n_test)):
            region_full[j,i] = conformal_full(y,x,x_test[i:i+1],alpha,y_plot,C,seed)
            coverage_full[j,i] = region_full[j,i,np.argmin(np.abs(y_test[i]-y_plot))]
            length_full[j,i] =np.sum(region_full[j,i])*dy
        end = time.time()
        times_full[j]= end - start
        
        #Bayes
        start = time.time()

        @jit #normal cdf from posterior samples
        def normal_likelihood_cdf(y,x):
            return norm.cdf(y,loc =jnp.dot(beta_post[j],x.transpose())+ intercept_post[j],scale = sigma_post[j]) #compute likelihood samples

        #Precompute cdfs
        cdf_test =  normal_likelihood_cdf(y_plot.reshape(-1,1,1),x_test)

        for i in (range(n_test)):
            band_bayes[j,i] = bmcmc.compute_bayes_band_MCMC(alpha,y_plot,cdf_test[:,:,i])
            coverage_bayes[j,i] = (y_test[i] >=band_bayes[j,i,0])&(y_test[i] <=band_bayes[j,i,1])
            length_bayes[j,i] = np.abs(band_bayes[j,i,1]- band_bayes[j,i,0])
        end = time.time()
        times_bayes[j] = end - start


        #Conformal Bayes
        start = time.time()
        @jit #normal loglik from posterior samples
        def normal_loglikelihood(y,x):
            return norm.logpdf(y,loc = jnp.dot(beta_post[j],x.transpose())+ intercept_post[j],scale = sigma_post[j]) #compute likelihood samples

        logp_samp_n = normal_loglikelihood(y,x)
        logwjk = normal_loglikelihood(y_plot.reshape(-1,1,1),x_test)
        logwjk_test = normal_loglikelihood(y_test,x_test).reshape(1,-1,n_test)

        for i in (range(n_test)):
            region_cb[j,i] = cb.compute_cb_region_IS(alpha,logp_samp_n,logwjk[:,:,i])
            coverage_cb[j,i] = region_cb[j,i,np.argmin(np.abs(y_test[i]-y_plot))] #grid coverage
            length_cb[j,i] = np.sum(region_cb[j,i])*dy
        end = time.time()
        times_cb[j] = end - start

        #compute exact coverage to avoid grid effects
        for i in (range(n_test)):
            coverage_cb_exact[j,i] = cb.compute_cb_region_IS(alpha,logp_samp_n,logwjk_test[:,:,i]) #exact coverage
    

    # #Save regions (need to update)
    np.save("results/region_cb_sparsereg_{}".format(suffix),region_cb)
    np.save("results/band_bayes_sparsereg_{}".format(suffix),band_bayes)
    np.save("results/band_split_sparsereg_{}".format(suffix),band_split)
    np.save("results/region_full_sparsereg_{}".format(suffix),band_split)

    np.save("results/coverage_cb_sparsereg_{}".format(suffix),coverage_cb)
    np.save("results/coverage_cb_exact_sparsereg_{}".format(suffix),coverage_cb_exact)
    np.save("results/coverage_bayes_sparsereg_{}".format(suffix),coverage_bayes)
    np.save("results/coverage_split_sparsereg_{}".format(suffix),coverage_split)
    np.save("results/coverage_full_sparsereg_{}".format(suffix),coverage_full)

    np.save("results/length_cb_sparsereg_{}".format(suffix),length_cb)
    np.save("results/length_bayes_sparsereg_{}".format(suffix),length_bayes)
    np.save("results/length_split_sparsereg_{}".format(suffix),length_split)
    np.save("results/length_full_sparsereg_{}".format(suffix),length_full)

    np.save("results/times_cb_sparsereg_{}".format(suffix),times_cb)
    np.save("results/times_bayes_sparsereg_{}".format(suffix),times_bayes)
    np.save("results/times_split_sparsereg_{}".format(suffix),times_split)
    np.save("results/times_full_sparsereg_{}".format(suffix),times_full)
