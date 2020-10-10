###########################################################################
# Imports
###########################################################################
import numpy as np
import os
import my_util

###########################################################################
# This function implements MLE for Bernoulli density function and cross-
# validates against given cross-validation data and chooses the best prior
# which gives lowest error rate on cross-validation data. It also prints
# error rate Vs. prior hyper-parameter table against cross-validation
# dataset
###########################################################################
def Bayes_Learning(t_data, v_data):
    ###########################################################################
    # Check if input files are present in the location
    ###########################################################################
    if not os.path.isfile(t_data):
        print("Training data file can't be located")
        return None, None, None, None
    if not os.path.isfile(v_data):
        print("Validation data file can't be located")
        return None, None, None, None
    ###########################################################################
    # Load training data and extract information
    ###########################################################################
    trn_data = np.loadtxt(t_data)
    N, D = trn_data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D -= 1
    # Separate samples corresponding to each class
    data_C_1 = trn_data[trn_data[:, D] == 1.]
    data_C_2 = trn_data[trn_data[:, D] == 2.]
    # Get number of samples corresponding to each class
    N_C_1 = np.size(data_C_1, 0)
    N_C_2 = np.size(data_C_2, 0)
    ###########################################################################
    # MLE for D-dimensional Bernoulli density
    ###########################################################################
    # Count number of 1's in each feature across all N samples and divide it by N to get (1-p_ij)
    # as p_ij denotes P(x_j=0|C_i)
    p_1j = 1. - ((np.count_nonzero(data_C_1[:, 0:D], axis=0))/N_C_1)
    p_2j = 1. - ((np.count_nonzero(data_C_2[:, 0:D], axis=0))/N_C_2)
    ###########################################################################
    # Create a vector of priors corresponding to each sigma
    # in {-5,-4,...,0,1,...,5}
    ###########################################################################
    start_index = -5
    end_index = 5
    sigma = np.linspace(start_index, end_index, end_index-start_index+1)
    ###########################################################################
    # Calculate priors for each sigma values
    ###########################################################################
    p_c1 = 1./(1.+np.exp(((-1.)*sigma)))
    #p_c2 = 1-p_c1
    ###########################################################################
    # Load validation data and extract information
    ###########################################################################
    val_data = np.loadtxt(v_data)
    N_val, D_val = val_data.shape
    # Last column represents class label, therefore feature dimension is D_val-1
    D_val -= 1
    ###########################################################################
    # Classify validation set data: Whole data set
    ###########################################################################
    prediction = np.empty((N_val,) + p_c1.shape)
    for prior_index in range(len(p_c1)):
        for sample_index in range(N_val):
            prediction[sample_index, prior_index] = my_util.classify(val_data[sample_index, 0:D], p_1j, p_2j, p_c1[prior_index], (1. - p_c1[prior_index]))
    ###########################################################################
    # Cross validation performance check: Whole data set
    ###########################################################################
    err = np.empty(p_c1.shape)
    for prior_index in range(len(p_c1)):
        err[prior_index] = (np.sum(prediction[:, prior_index] != val_data[:, D_val]))*(100./N_val)
    my_util.print_error_table("Error Rate-Prior Table: Validation dataset", sigma, p_c1, err)

    ###########################################################################
    # Get best prior corresponding to minimum error rate
    ###########################################################################
    best_prior_index = np.argmin(err)
    # print(sigma[best_prior_index])
    return p_1j, p_2j, p_c1[best_prior_index], (1.-p_c1[best_prior_index])