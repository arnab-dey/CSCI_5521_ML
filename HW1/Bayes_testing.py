###########################################################################
# Imports
###########################################################################
import numpy as np
import os
import my_util


###########################################################################
# Function Definition
###########################################################################
###########################################################################
# This function classifies samples based on prior and Bernoulli parameter
# and prints the error rate
###########################################################################
def Bayes_Testing(t_data, p1, p2, p_c1, p_c2):
    ###########################################################################
    # Check if input file is present in the location
    ###########################################################################
    if not os.path.isfile(t_data):
        print("Test data file can't be located")
        return
    ###########################################################################
    # Load test data and extract information
    ###########################################################################
    test_data = np.loadtxt(t_data)
    N_test, D_test = test_data.shape
    # Last column represents class label, therefore feature dimension is D_test-1
    D_test -= 1

    ###########################################################################
    # Classify test set data
    ###########################################################################
    prediction = np.empty((N_test,))
    for sample_index in range(N_test):
        prediction[sample_index] = my_util.classify(test_data[sample_index, 0:D_test], p1, p2, p_c1, p_c2)
    ###########################################################################
    # test data performance check
    ###########################################################################
    err = (np.sum(prediction != test_data[:, D_test]))*(100./N_test)
    my_util.print_error_table("Error Rate: Test dataset", None, p_c1, err)
