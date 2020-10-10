###########################################################################
# Imports
###########################################################################
import numpy as np


###########################################################################
# Function Definitions
###########################################################################
###########################################################################
# This function implements the discriminant function logic and
# classifies sample data according to the MLEs and priors given as input
###########################################################################
def classify(sample, p1j, p2j, p_c1, p_c2):
    ###########################################################################
    # Discriminant function implementation
    ###########################################################################
    log_p1j = np.log(p1j)
    log_p2j = np.log(p2j)
    log_p1j_bar = np.log((1.-p1j))
    log_p2j_bar = np.log((1.-p2j))
    g_1 = np.sum(((1.-sample)*(log_p1j)) + (sample*(log_p1j_bar))) + np.log(p_c1)
    g_2 = np.sum(((1.-sample)*(log_p2j)) + (sample*(log_p2j_bar))) + np.log(p_c2)
    return 1. if g_1 >= g_2 else 2.

###########################################################################
# This function prints error rate table corresponding to prior values
###########################################################################
def print_error_table(table_heading, sigma=None, p_c1=None, error_arr=None):
    if p_c1 is None or error_arr is None:
        print("Invalid arguments")
        return
    print(table_heading)
    if type(p_c1) is np.ndarray:
        for prior_index in range(len(sigma)):
            if sigma is not None:
                print("sigma = ", sigma[prior_index], ", P(C_1) = ", p_c1[prior_index], ", Error Rate(%) = ", error_arr[prior_index])
            else:
                print("P(C_1) = ", p_c1[prior_index], ", Error Rate(%) = ", error_arr[prior_index])
    else:
        if sigma is not None:
            print("sigma = ", sigma, ", P(C_1) = ", p_c1, ", Error Rate(%) = ", error_arr)
        else:
            print("P(C_1) = ", p_c1, ", Error Rate(%) = ", error_arr)
    return
