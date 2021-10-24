from run_scripts.run_sparsereg_mcmc import run_sparsereg_mcmc
from run_scripts.run_ridgereg_mcmc import run_ridgereg_mcmc
from run_scripts.run_sparseclass_mcmc import run_sparseclass_mcmc
from run_scripts.run_hier_mcmc import run_hier_mcmc

from run_scripts.run_sparsereg_conformal import run_sparsereg_conformal
from run_scripts.run_ridgereg_conformal import run_ridgereg_conformal
from run_scripts.run_sparseclass_conformal import run_sparseclass_conformal
from run_scripts.run_hier_conformal import run_hier_conformal
from run_scripts.run_hier_conformal_split import run_hier_conformal_split

# Run MCMC #
#Sparse reg: Diabetes#
# run_sparsereg_mcmc('diabetes', misspec = False)
# run_sparsereg_mcmc('diabetes', misspec = True)
# run_ridgereg_mcmc('diabetes', misspec = False)
# run_ridgereg_mcmc('diabetes', misspec = True)

# #Sparse reg: Boston Housing#
# run_sparsereg_mcmc('boston', misspec = False)
# run_sparsereg_mcmc('boston', misspec = True)

# #Sparse class: Breast Cancer#
# run_sparseclass_mcmc('breast')

# #Sparse class: Parkinsons#
# run_sparseclass_mcmc('parkinsons')

# #Hier: Simulated#
# run_hier_mcmc('sim', misspec = False)
# run_hier_mcmc('sim',misspec = True)

# #Hier: Radon#
# run_hier_mcmc('radon')

# # Run Conformal Bayes #
# #Sparse reg: Diabetes#
# run_sparsereg_conformal('diabetes', misspec = False)
# run_sparsereg_conformal('diabetes', misspec = True)
# run_ridgereg_conformal('diabetes', misspec = False) # With normal prior
# run_ridgereg_conformal('diabetes', misspec = True)

# #Sparse reg: Boston Housing#
# run_sparsereg_conformal('boston', misspec = False)
# run_sparsereg_conformal('boston', misspec = True)

# #Sparse class: Breast Cancer#
# run_sparseclass_conformal('breast')

# #Sparse class: Parkinsons#
# run_sparseclass_conformal('parkinsons')

# #Hier: Simulated#
# run_hier_conformal('sim', misspec = False)
# run_hier_conformal('sim',misspec = True)
run_hier_conformal_split('sim', misspec = False)
run_hier_conformal_split('sim',misspec = True)

#Hier: Radon#
#run_hier_conformal('radon',misspec = False)
#run_hier_conformal('radon',misspec = False) #run twice to handle compilation overhead (not as noticeable for 50 repeats)