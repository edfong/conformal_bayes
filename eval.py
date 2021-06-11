import numpy as np

examples = ['sparsereg_diabetes','sparsereg_diabetes_misspec','ridgereg_diabetes','ridgereg_diabetes_misspec','sparsereg_boston','sparsereg_boston_misspec'\
,'sparseclass_breast', 'sparseclass_parkinsons','hier_sim', 'hier_sim_misspec', 'hier_radon']

methods_tot = ['bayes', 'cb', 'split', 'full']


# Report MCMC times #
for example in examples:
    suffix = example
    times = np.load("samples/times_{}.npy".format(suffix))
    rep = np.shape(times)[0]
    print("{} MCMC time: {:.3f} ({:.3f})".format(suffix,np.mean(times), np.std(times)/np.sqrt(rep)))
print()

# Report Conformal results#
for example in examples:
    if example in ['hier_sim', 'hier_sim_misspec', 'hier_radon']:
        methods = methods_tot[0:2]
    else:
        methods = methods_tot

    print('EXAMPLE: {}'.format(example))
    for method in methods:
        suffix = method +'_' + example

        # Coverage
        coverage = np.mean(np.load("results/coverage_{}.npy".format(suffix)),axis = 1) #take mean over test values
        rep = np.shape(coverage)[0]
        mean = np.mean(coverage)
        se = np.std(coverage)/np.sqrt(rep)
        print("{} coverage is {:.3f} ({:.3f})".format(method, mean,se)) 

        # Return exact coverage if cb
        if method == 'cb' and (example not in ['sparseclass_breast', 'sparseclass_parkinsons']):
            suffix_ex = method +'_exact_' + example
            coverage = np.mean(np.load("results/coverage_{}.npy".format(suffix_ex)),axis = 1) #take mean over test values
            rep = np.shape(coverage)[0]
            mean = np.mean(coverage)
            se = np.std(coverage)/np.sqrt(rep)
            print("{} exact coverage is {:.3f} ({:.3f})".format(method, mean,se))
    print()

    for method in methods:
        suffix = method +'_' + example
        # Length
        length = np.mean(np.load("results/length_{}.npy".format(suffix)),axis = 1)
        rep = np.shape(length)[0]
        mean = np.mean(length)
        se = np.std(length)/np.sqrt(rep)
        print("{} length is {:.2f} ({:.2f})".format(method, mean,se)) # Times
    print()

    for method in methods:
        suffix = method +'_' + example
        # Length
        times = np.load("results/times_{}.npy".format(suffix))
        rep = np.shape(times)[0]
        mean = np.mean(times)
        se = np.std(times)/np.sqrt(rep)
        print("{} times is {:.3f} ({:.3f})".format(method, mean,se)) # Times
    print()

    #print misclassification/both/empty
    if example in ['sparseclass_breast', 'sparseclass_parkinsons']:
        for method in methods[0:2]:
            suffix = method +'_' + example
            coverage = np.load("results/coverage_{}.npy".format(suffix))
            length = np.load("results/length_{}.npy".format(suffix))
            rep = np.shape(coverage)[0]
            n_tot = np.sum(length== 1,axis = 1) 
            n_misclass = np.sum(np.logical_and(length ==1, coverage == 0),axis = 1)
            misclass_rate = n_misclass/n_tot
            both_rate = np.mean(length == 2, axis = 1)
            empty_rate = np.mean(length == 0, axis = 1)

            print('{} misclasification rate is {:.3f} ({:.3f})'.format(method,np.mean(misclass_rate), np.std(misclass_rate)/np.sqrt(rep)))
            print('{} both rate is {:.3f} ({:.3f})'.format(method,np.mean(both_rate), np.std(both_rate)/np.sqrt(rep)))
            print('{} empty rate is {:.3f} ({:.3f})'.format(method,np.mean(empty_rate), np.std(empty_rate)/np.sqrt(rep)))




    #print per group for hierarchical
    if example in ['hier_sim', 'hier_sim_misspec']:
        for method in methods:
            suffix = method +'_grp_' + example
            print('Method is {}'.format(method))
            #Group coverage
            coverage_grp = np.load("results/coverage_{}.npy".format(suffix))
            rep = np.shape(coverage_grp)[1]
            K = np.shape(coverage_grp)[0]
            for k in range(K):
                mean = np.mean(coverage_grp[k])
                se = np.std(coverage_grp[k])/np.sqrt(rep)
                print("Group {} coverage: {:.3f} ({:.3f})".format(k,mean,se))

            # Return exact coverage if cb
            if method == 'cb':
                suffix_ex = method +'_exact_grp_' + example
                coverage_grp = np.load("results/coverage_{}.npy".format(suffix_ex))
                for k in range(K):
                    mean = np.mean(coverage_grp[k])
                    se = np.std(coverage_grp[k])/np.sqrt(rep)
                    print("Group {} exact coverage is {:.3f} ({:.3f})".format(example, mean,se))
            print()

        for method in methods:
            suffix = method +'_grp_' + example
            #Group length
            length_grp = np.load("results/length_{}.npy".format(suffix))
            for k in range(K):
                mean = np.mean(length_grp[k])
                se = np.std(length_grp[k])/np.sqrt(rep)
                print("Group {} length: {:.2f} ({:.2f})".format(k,mean,se))
            print()
    print()