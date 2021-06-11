from sklearn.datasets import load_boston,load_diabetes,load_breast_cancer
from sklearn.model_selection import train_test_split
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy as sp

## Sparse Regression ##
#Well specified?
def load_traintest_sparsereg(train_frac, dataset,seed):
    #Load dataset
    if dataset =="diabetes":
        x,y = load_diabetes(return_X_y = True)
    elif dataset =="boston":
        x,y = load_boston(return_X_y = True)
    else:
        print('Invalid dataset')
        return

    n = np.shape(x)[0]
    d = np.shape(x)[1]

    #Standardize beforehand (for validity)
    x = (x - np.mean(x,axis = 0))/np.std(x,axis = 0)
    y = (y - np.mean(y))/np.std(y)

    #Train test split
    ind_train, ind_test = train_test_split(np.arange(n), train_size = int(train_frac*n),random_state = seed)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    y_plot = np.linspace(np.min(y_train) - 2, np.max(y_train) + 2,100)
    
    return x_train,y_train,x_test,y_test,y_plot,n,d

## Sparse Classification ##
# Load data
def load_traintest_sparseclass(train_frac,dataset, seed):
    #Load dataset
    if dataset =="breast":
        x,y = load_breast_cancer(return_X_y = True)
    elif dataset == "parkinsons":
        data = pd.read_csv('data/parkinsons.data')
        data[data == '?']= np.nan
        data.dropna(axis = 0,inplace = True)
        y = data['status'].values #convert strings to integer
        x = data.drop(columns = ['name','status']).values
    else:
        print('Invalid dataset')
        return

    n = np.shape(x)[0]
    d = np.shape(x)[1]

    #Standardize beforehand (for validity)
    x = (x - np.mean(x,axis = 0))/np.std(x,axis = 0)

    #Train test split
    ind_train, ind_test = train_test_split(np.arange(n), train_size = int(train_frac*n),random_state = seed)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    y_plot = np.array([0,1])
    
    return x_train,y_train,x_test,y_test,y_plot,n,d


## Hierarchical Datasets ##
#Simulate data 
def gen_data_hier(n,p,n_test,seed,K,misspec = False): #n and n_test is number per group
    #Generate groups first
    theta = np.zeros(p)

    #Generate K beta_values (fixed random values)
    np.random.seed(24)
    beta_true = np.random.randn(K,p) + theta.reshape(1,-1)
    sigma_true = np.random.exponential(size = K, scale = 1)
    
    #Training data
    np.random.seed(seed) #try new seed
    x = np.zeros((n*K,p+1))
    y = np.zeros(n*K)

    for k in range(K):
        if misspec == True:
            #eps = sp.stats.skewnorm.rvs(a=5,size = n)
            eps = np.random.randn(n)*sigma_true[k]
        else:
            eps = np.random.randn(n) 
        x[k*n:(k+1)*n] = np.concatenate((np.random.randn(n,p),k*np.ones((n,1))),axis = 1) #Append group index to last dimension
        y[k*n:(k+1)*n] = np.dot(x[k*n:(k+1)*n,0:p],beta_true[k]) + eps
    
    #Test data
    x_test = np.zeros((n_test*(K),p+1))
    y_test = np.zeros(n_test*(K))

    for k in range(K): 
        if misspec == True:
            #eps_test = sp.stats.skewnorm.rvs(a=5,size = n_test)
            eps_test = np.random.randn(n_test)*sigma_true[k]
        else:
            eps_test = np.random.randn(n_test) 
        x_test[k*n_test:(k+1)*n_test] = np.concatenate((np.random.randn(n_test,p),k*np.ones((n_test,1))),axis = 1) #Append group index to last dimension
        y_test[k*n_test:(k+1)*n_test] = np.dot(x_test[k*n_test:(k+1)*n_test,0:p],beta_true[k]) + eps_test
    
    y_plot = np.linspace(-10,10,100)
    return y,x,y_test,x_test,beta_true,sigma_true,y_plot

# Load Radon (Minnesota) dataset, based on https://docs.pymc.io/notebooks/multilevel_modeling.html
def load_traintest_hier(train_frac,dataset, seed):
    #Load dataset
    if dataset =="radon":
        # Import radon data
        srrs2 = pd.read_csv("./data/srrs2.dat")
        srrs2.columns = srrs2.columns.map(str.strip)
        srrs_mn = srrs2[srrs2.state == "MN"].copy()

        srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
        cty = pd.read_csv("./data/cty.dat")
        cty_mn = cty[cty.st == "MN"].copy()
        cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

        srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
        srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
        u = np.log(srrs_mn.Uppm).unique()

        n = len(srrs_mn)

        srrs_mn.county = srrs_mn.county.map(str.strip)
        mn_counties = srrs_mn.county.unique()
        counties = len(mn_counties)
        county_lookup = dict(zip(mn_counties, range(counties)))

        county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
        radon = srrs_mn.activity
        srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
        floor = srrs_mn.floor.values

        #Preprocess
        x = np.zeros((n,2))
        x[:,0] = floor
        x[:,1]= county
        x = np.array(x, dtype = 'int')
        y = np.array(log_radon)
        
    else:
        print('Invalid dataset')
        return

    n = np.shape(x)[0]
    d = np.shape(x)[1]

    #Train test split
    if train_frac ==1.:
        ind_train = np.arange(n)
        ind_test = np.array([],dtype = 'int')
    else:
        ind_train, ind_test = train_test_split(np.arange(n), train_size = int(train_frac*n),random_state = seed,stratify = x[:,1])
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    y_plot = np.linspace(-6,6,100)
    
    return x_train,y_train,x_test,y_test,y_plot,n,d
## ##