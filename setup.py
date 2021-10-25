from setuptools import setup,find_packages
#May need to install Pystan separately with pip
setup(name='conformal_bayes',
      version='0.1.0',
      description='Conformal Bayes Predictive Intervals',
      license='BSD 3-Clause',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.20.3',
          'scipy>=1.6.3',
          'scikit-learn>=0.24.2',
          'pandas',
          'matplotlib',
          'seaborn',
          'joblib',
          'tqdm',
          'jax>=0.2.13',
          'jaxlib>=0.1.66',
          'pydataset',
          'xlrd',
          'pymc3>=3.11.2',
          'Theano-PyMC>=1.1.2'
      ],
      include_package_data=True,
      python_requires='>=3.7.5'
      )

#For GPU
#pip install --upgrade jax jaxlib==0.1.66+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
#Only tested with Pyton 3.7.5