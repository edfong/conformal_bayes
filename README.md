# Conformal Bayesian Computation

This repository is the official implementation of [Conformal Bayesian Computation](https://arxiv.org/abs/2106.06137). 

## Requirements

To install requirements and the `conformal_bayes` package, run the following in the main folder:

```setup
python3 setup.py install
```

If the above does not work, then please run the following instead:

```
pip install .
```

This should install all dependencies needed. We have only tested this on Python 3.7.5, and for reproducibility we have used `jax (0.2.13)` and `jaxlib (0.1.66)`. We recommend creating a new virtualenv/conda environment before installing the package, and using the latest verson of `setuptools` or `pip`. 

We have included PyMC3 as a dependency for `conformal_bayes` for convenience, but computing conformal Bayesian intervals only requires [JAX](https://github.com/google/jax). If you pip fails to install PyMC3, please take a look at the installation instructions in the sidebar [here](https://github.com/pymc-devs/pymc3/wiki). If PyMC3 returns errors related to multiprocessing, try seting `chains = 1` in the MCMC scripts (with `_mcmc.py`).

The required datasets are self-contained in this repo. We have provided the [Radon](http://www.stat.columbia.edu/~gelman/arm/examples/radon/) dataset (Gelman and Hill (2006)) and [Parkinsons](https://archive.ics.uci.edu/ml/datasets/parkinsons) dataset (Little et al. (2008)) in the `./data` folder, and other datasets are already available in `sklearn` or are simulated. 

## Training

To carry out all experiments in the paper, run this command:

```train
python3 train.py
```
This will run all the MCMC examples and compute all intervals over 50 repetitions by calling functions in `./run_expts`. The functions in this folder are split between MCMC scripts (with `_mcmc.py`) and conformal scripts (with `_conformal.py`).

The posterior samples will be saved in  `./samples` and intervals/coverage/timings will be saved in `./results`.


## Evaluation

To load and print all the coverage/length/timing results, run:

```eval
python3 eval.py
```
This script will load the results in the `./results` folder, and print them in a similar format to the tables in the paper. Note that the evaluation (such as computing coverage/length) is actually computed in the `_conformal.py` functions.

## Pre-trained Models

Note that running MCMC will take a significant amount of time, but we are only able to provide posterior samples from the diabetes example to keep within file size limits. However, we have included the results folder so running the evaluation script will still work without fitting the models.

## Notebooks

We also provide a few Jupyter notebooks for producing the plots, which rely on the `./results` folder.


## References
Gelman, A., & Hill, J. (2006). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.

Little, M., McSharry, P., Hunter, E., Spielman, J., & Ramig, L. (2008). Suitability of dysphonia measurements for telemonitoring of Parkinson’s disease. Nature Precedings, 1-1.
