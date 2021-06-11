import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
import jax.scipy as jsp
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

#import from cb package
from run_scripts.load_data import load_traintest_sparseclass
from conformal_bayes import conformal_Bayes_functions as cb
from conformal_bayes import Bayes_MCMC_functions as bmcmc

#Define baselines
#Split Method
def conformal_split(alpha,y,x,x_test,seed = 100):
    n = np.shape(y)[0]
    n_test = np.shape(x_test)[0]
    #Fit lasso to training set
    n_train = int(n/2)
    ls = LogisticRegressionCV(penalty = 'l1', solver  = 'liblinear', cv = 5, random_state = seed)
    ls.fit(x[0:n_train],y[0:n_train])
    resid = ls.predict_proba(x[n_train:])[:,1]
    resid[y[n_train:]==0] =  1- resid[y[n_train:]==0]
    resid = -np.log(np.clip(resid,1e-6,1-1e-6)) #clip for numerical stability
    k = int(np.ceil((n/2 + 1)*(1-alpha)))
    d = np.sort(resid)[k]    
    
    logp_test = -np.log(np.clip(ls.predict_proba(x_test),1e-6,1-1e-6))
    region_split = logp_test <= d
    
    return region_split

#Full Method
def conformal_full(alpha,y,x,x_test,C,seed = 100):
    n = np.shape(y)[0]
    rank_cp = np.zeros(2)
    for y_new in (0,1):
        x_aug = np.concatenate((x,x_test),axis = 0)
        y_aug = np.append(y,y_new)
        ls = LogisticRegression(penalty = 'l1', solver  = 'liblinear', C = C, random_state = seed)
        ls.fit(x_aug,y_aug)
        resid = ls.predict_proba(x_aug)[:,1]
        resid[y_aug==0] =  1- resid[y_aug==0]
        resid = -np.log(resid)
        rank_cp[y_new] = np.sum(resid>=resid[-1])/(n+1)
    region_full = rank_cp > alpha 
    return region_full

#Main run function for sparse classification
def run_sparseclass_conformal(dataset):

    #Compute intervals
    #Load posterior samples
    beta_post = jnp.load("samples/beta_post_sparseclass_{}.npy".format(dataset))
    intercept_post = jnp.load("samples/intercept_post_sparseclass_{}.npy".format(dataset))


    #Initialize
    train_frac = 0.7
    x,y,x_test,y_test,y_plot,n,d = load_traintest_sparseclass(train_frac,dataset,100)

    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep,n_test))
    coverage_bayes = np.zeros((rep,n_test))
    coverage_split = np.zeros((rep,n_test))
    coverage_full = np.zeros((rep,n_test))

    length_cb = np.zeros((rep,n_test))
    length_bayes = np.zeros((rep,n_test))
    length_split = np.zeros((rep,n_test))
    length_full= np.zeros((rep,n_test))
    
    p_bayes = np.zeros((rep,n_test))
    region_bayes = np.zeros((rep,n_test,2))
    region_cb = np.zeros((rep,n_test,2))
    region_split = np.zeros((rep,n_test,2))
    region_full = np.zeros((rep,n_test,2))

    times_bayes = np.zeros(rep)
    times_cb = np.zeros(rep)
    times_split = np.zeros(rep)
    times_full = np.zeros(rep)


    for j in tqdm(range(rep)):
        seed = 100 + j

        #load data
        x,y,x_test,y_test,y_plot,n,d = load_traintest_sparseclass(train_frac,dataset,seed)

        #Split conformal method
        start = time.time() 
        region_split[j] = conformal_split(alpha,y,x,x_test,seed)
        for i in (range(n_test)):
            coverage_split[j,i] = region_split[j,i,np.argmin(np.abs(y_test[i]-y_plot))]
            length_split[j,i] = np.sum(region_split[j,i])
        end = time.time()
        times_split[j]= end-start        


        #Full conformal method
        start = time.time() 
        C = 1.
        for i in (range(n_test)):
            region_full[j,i] = conformal_full(alpha,y,x,x_test[i:i+1],C,seed)
            coverage_full[j,i] = region_full[j,i,np.argmin(np.abs(y_test[i]-y_plot))]
            length_full[j,i] = np.sum(region_full[j,i])
        end = time.time()
        times_full[j]= end-start

        #Bayes
        start = time.time()
        @jit
        def logistic_loglikelihood(y,x):
            eta = (jnp.dot(beta_post[j],x.transpose())+intercept_post[j])
            B = np.shape(eta)[0]
            n = np.shape(eta)[1]
            eta = eta.reshape(B,n,1)
            temp0 = np.zeros((B,n,1))
            logp = -jsp.special.logsumexp(jnp.concatenate((temp0,-eta),axis = 2),axis = 2) #numerically stable
            log1p = -jsp.special.logsumexp(jnp.concatenate((temp0,eta),axis = 2),axis = 2)
            return y*logp + (1-y)*log1p #compute likelihood samples
        
        for i in (range(n_test)):
            p_bayes[j,i] = jnp.mean(jnp.exp(logistic_loglikelihood(1,x_test[i:i+1])))
            #Compute region from p_bayes
            if p_bayes[j,i] >(1-alpha): #only y = 1
                region_bayes[j,i] = np.array([0,1])
            elif (1-p_bayes[j,i]) >(1-alpha):  #only y = 0
                region_bayes[j,i] = np.array([1,0])
            else:
                region_bayes[j,i] = np.array([1,1])
            coverage_bayes[j,i] = region_bayes[j,i,np.argmin(np.abs(y_test[i]-y_plot))]
            length_bayes[j,i] = np.sum(region_bayes[j,i])
        end = time.time()
        times_bayes[j]= end-start

        #Conformal Bayes
        start = time.time()
        logp_samp_n = logistic_loglikelihood(y,x)
        logwjk = logistic_loglikelihood(y_plot.reshape(-1,1,1),x_test)
        #conformal
        for i in (range(n_test)):
            region_cb[j,i] = cb.compute_cb_region_IS(alpha,logp_samp_n,logwjk[:,:,i])
            coverage_cb[j,i] = region_cb[j,i,np.argmin(np.abs(y_test[i]-y_plot))]
            length_cb[j,i] = np.sum(region_cb[j,i])
        end = time.time()
        times_cb[j] = end-start


    #Save regions (need to update)
    np.save("results/p_bayes_sparseclass_{}".format(dataset),p_bayes)
    np.save("results/region_bayes_sparseclass_{}".format(dataset),region_bayes)
    np.save("results/region_cb_sparseclass_{}".format(dataset),region_cb)
    np.save("results/region_split_sparseclass_{}".format(dataset),region_split)
    np.save("results/region_full_sparseclass_{}".format(dataset),region_full)

    np.save("results/coverage_bayes_sparseclass_{}".format(dataset),coverage_bayes)
    np.save("results/coverage_cb_sparseclass_{}".format(dataset),coverage_cb)
    np.save("results/coverage_split_sparseclass_{}".format(dataset),coverage_split)
    np.save("results/coverage_full_sparseclass_{}".format(dataset),coverage_full)
    
    np.save("results/length_bayes_sparseclass_{}".format(dataset),length_bayes)
    np.save("results/length_cb_sparseclass_{}".format(dataset),length_cb)
    np.save("results/length_split_sparseclass_{}".format(dataset),length_split)
    np.save("results/length_full_sparseclass_{}".format(dataset),length_full)

    np.save("results/times_bayes_sparseclass_{}".format(dataset),times_bayes)
    np.save("results/times_cb_sparseclass_{}".format(dataset),times_cb)
    np.save("results/times_split_sparseclass_{}".format(dataset),times_split)
    np.save("results/times_full_sparseclass_{}".format(dataset),times_full)


