import jax.numpy as jnp
from jax import jit
from functools import partial 
import numpy as np
import scipy as sp
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from jax.ops import index_update

## CONFORMAL FROM MCMC SAMPLES ##
### JAX IMPLEMENTATION
@jit #compute rank (unnormalized by n+1)
def compute_rank_IS(logp_samp_n,logwjk):
    n= jnp.shape(logp_samp_n)[1] #logp_samp_n is B x n
    n_plot = jnp.shape(logwjk)[0]
    rank_cp = jnp.zeros(n_plot)
    
    #compute importance sampling weights and normalizing
    wjk = jnp.exp(logwjk)
    Zjk = jnp.sum(wjk,axis = 1).reshape(-1,1)
    
    #compute predictives for y_i,x_i and y_new,x_n+1
    p_cp = jnp.dot(wjk/Zjk, jnp.exp(logp_samp_n))
    p_new = jnp.sum(wjk**2,axis = 1).reshape(-1,1)/Zjk

    #compute nonconformity score and sort
    pred_tot = jnp.concatenate((p_cp,p_new),axis = 1)
    rank_cp = np.sum(pred_tot <= pred_tot[:,-1].reshape(-1,1),axis = 1)
    return rank_cp


#compute region of grid which is in confidence set
@jit
def compute_cb_region_IS(alpha,logp_samp_n,logwjk): #assumes they are connected
    n= jnp.shape(logp_samp_n)[1]#logp_samp_n is B x n
    rank_cp = compute_rank_IS(logp_samp_n,logwjk)
    region_true =rank_cp> alpha*(n+1)
    return region_true
## ##

## DIAGNOSE IMPORTANCE WEIGHTS ##
@jit #compute ESS/var
def diagnose_is_weights(logp_samp_n,logwjk):
    n= jnp.shape(logp_samp_n)[1] #logp_samp_n is B x n
    n_plot = jnp.shape(logwjk)[0]
    rank_cp = jnp.zeros(n_plot)
    
    #compute importance sampling weights and normalizing
    logwjk = logwjk.reshape(n_plot,-1, 1)
    logZjk = logsumexp(logwjk,axis = 1)
    
    #compute predictives for y_i,x_i and y_new,x_n+1
    logp_new = logsumexp(2*logwjk,axis = 1)-logZjk 

    #compute ESS
    wjk = jnp.exp(logwjk - logZjk.reshape(-1,1,1))
    ESS = 1/jnp.sum(wjk**2,axis = 1)

    #compute variance for p_new
    var = np.sum(wjk**2*(wjk - jnp.exp(logp_new).reshape(-1,1,1))**2,axis = 1)
    return ESS, var
###
