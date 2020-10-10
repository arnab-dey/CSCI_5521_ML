###########################################################################
# Imports
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
###########################################################################
# Function Definitions
###########################################################################
def myPCA(data, n_prin_comp):
    ###########################################################################
    # Extract information data
    ###########################################################################
    N, D = data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D -= 1
    ###########################################################################
    # MLE of mean
    ###########################################################################
    mu = np.sum(data[:, 0:D], axis=0)/N
    ###########################################################################
    # MLE of variance
    ###########################################################################
    diff_from_mean = data[:, 0:D] - mu
    S = diff_from_mean.T @ diff_from_mean/N
    ###########################################################################
    # Compute eigen-values and eigen-vectors of covariance matrix and sort it
    ###########################################################################
    eps = 1e-8
    # As S is symmetric and positive semi-definite, we expect e-values to be
    # greater than or equal to 0. Therefore, we can use np.eigh and arrange
    # e-values based on their values. No need to take absolute value
    e_val, e_vec = np.linalg.eigh(S)
    e_val[np.abs(e_val) < eps] = 0
    e_sort_index = np.argsort(e_val)[::-1]
    e_val = e_val[e_sort_index]
    e_vec = e_vec[:, e_sort_index]
    ###########################################################################
    # Compute proportion of variance
    ###########################################################################
    sum_e_val = np.sum(e_val)
    prop_var = np.cumsum(e_val)/sum_e_val
    ###########################################################################
    # Plot proportion of variance
    ###########################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle="-.", linewidth='0.5')
    ax.set_xlim(0, D+5)

    plt.plot(prop_var, 'k', marker='+')
    plt.title('Proportion of variance')
    plt.xlabel('Number of eigenvectors')
    plt.ylabel('Prop. of var.')
    plt.show()
    ###########################################################################
    # Choose minimum number of e-vec that explains 90% of variance
    ###########################################################################
    min_num_e_vec = (np.where(prop_var >= 0.9))[0][0]
    print("Minimum number of eigenvectors that explain 90% of the variance = ", min_num_e_vec)
    return e_vec[:, 0:n_prin_comp], e_val[0:n_prin_comp]



