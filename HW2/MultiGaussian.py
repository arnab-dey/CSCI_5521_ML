###########################################################################
# Imports
###########################################################################
import numpy as np
import os
###########################################################################
# Function Definitions
###########################################################################
###########################################################################
# This function implements the discriminant function logic and
# classifies sample data according to the MLEs and priors given as input
###########################################################################
def classify(sample, mu1, mu2, sigma1, sigma2, p_c1, p_c2):
    ###########################################################################
    # Discriminant function implementation
    ###########################################################################
    # print(np.linalg.det(sigma1))
    # print(np.linalg.det(sigma2))
    log_S_1 = np.log(np.linalg.det(sigma1))
    log_S_2 = np.log(np.linalg.det(sigma2))
    quad_term_1 = (sample-mu1) @ np.linalg.inv(sigma1) @ (sample-mu1).T
    quad_term_2 = (sample - mu2) @ np.linalg.inv(sigma2) @ (sample - mu2).T
    g_1 = -0.5*log_S_1 -0.5*quad_term_1 + np.log(p_c1)
    g_2 = -0.5*log_S_2 -0.5*quad_term_2 + np.log(p_c2)
    # log odds, if expanded, boils down to log(g_1)-log(g_2). Therefore, C1 is chosen if log odds is positive i.e.
    # g_1 >= g_2 or otherwise
    return 1. if g_1 >= g_2 else 2.
###########################################################################
# This function implements MLE for Bernoulli density function and cross-
# validates against given cross-validation data and chooses the best prior
# which gives lowest error rate on cross-validation data. It also prints
# error rate Vs. prior hyper-parameter table against cross-validation
# dataset
###########################################################################
def MultiGaussian(training_data, testing_data, Model):
    ###########################################################################
    # Check if input files are present in the location
    ###########################################################################
    if not os.path.isfile(training_data):
        print("Training data file can't be located")
        return None, None, None, None, None, None
    if not os.path.isfile(testing_data):
        print("testing data file can't be located")
        return None, None, None, None, None, None
    ###########################################################################
    # Load training data and extract information
    ###########################################################################
    trn_data = np.loadtxt(training_data, delimiter=',')
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
    # Estimation of priors
    ###########################################################################
    p_C_1 = N_C_1/N
    p_C_2 = N_C_2/N
    ###########################################################################
    # MLE of mean
    ###########################################################################
    mu_1 = np.sum(data_C_1[:, 0:D], axis=0)/N_C_1
    mu_2 = np.sum(data_C_2[:, 0:D], axis=0)/N_C_2
    ###########################################################################
    # MLE of variance based on the model number
    ###########################################################################
    diff_from_mean_C_1 = data_C_1[:, 0:D] - mu_1
    diff_from_mean_C_2 = data_C_2[:, 0:D] - mu_2
    # print(diff_from_mean_C_1)
    S_1_general = diff_from_mean_C_1.T @ diff_from_mean_C_1 / N_C_1
    S_2_general = diff_from_mean_C_2.T @ diff_from_mean_C_2 / N_C_2
    if Model == 1:
        S_1 = S_1_general
        S_2 = S_2_general
        print(np.linalg.det(S_1))
        print(np.linalg.det(S_2))
    elif Model == 2:
        # common covariance marix is the expectation of both S_1 and S_2
        S = p_C_1*S_1_general + p_C_2*S_2_general
        S_1 = S
        S_2 = S
    elif Model == 3:
        S_1 = np.diag(np.diag(S_1_general))
        S_2 = np.diag(np.diag(S_2_general))
    else:
        S_1 = S_1_general
        S_2 = S_2_general

    ###########################################################################
    # Print the learned parameters
    ###########################################################################
    print("Model No. ", Model)
    print("P(C_1) = ", p_C_1, " P(C_2) = ", p_C_2)
    print("mu_1 = ", mu_1)
    print("mu_2 = ", mu_2)
    print("Covariance matrix for C1 = ", S_1)
    print("Covariance matrix for C2 = ", S_2)
    ###########################################################################
    # Load test data and extract information
    ###########################################################################
    test_data = np.loadtxt(testing_data, delimiter=',')
    N_test, D_test = test_data.shape
    # Last column represents class label, therefore feature dimension is D_test-1
    D_test -= 1
    ###########################################################################
    # Classify test set data
    ###########################################################################
    prediction = np.empty((N_test,))
    for sample_index in range(N_test):
        prediction[sample_index] = classify(test_data[sample_index, 0:D_test], mu_1, mu_2, S_1, S_2, p_C_1, p_C_2)
    ###########################################################################
    # test data performance check
    ###########################################################################
    err = (np.sum(prediction != test_data[:, D_test])) * (100. / N_test)
    print("Error Rate for ", testing_data, " using model number ", Model, " = ", err)
    return p_C_1, p_C_2, mu_1, mu_2, S_1, S_2

