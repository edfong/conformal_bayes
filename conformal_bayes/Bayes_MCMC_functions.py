import jax.numpy as jnp
from jax import jit
from functools import partial 
import numpy as np
import scipy as sp
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from jax.ops import index_update

## BAYESIAN CREDIBLE INTERVALS FROM MCMC SAMPLES ##
#compute bayesian central 1-alpha credible interval
@jit
def compute_bayes_band_MCMC(alpha,y_plot,cdf_pred):
    cdf_pred = jnp.mean(cdf_pred,axis = 1)
    
    band_bayes = np.zeros(2)
    band_bayes = index_update(band_bayes,0, y_plot[jnp.argmin(jnp.abs(cdf_pred - alpha/2))])
    band_bayes =index_update(band_bayes,1,y_plot[jnp.argmin(jnp.abs(cdf_pred - (1-alpha/2)))])
    return(band_bayes)
