###########################################################################
# Imports
###########################################################################
import numpy as np
###########################################################################
# Function Definitions
###########################################################################
def myLDA(data, n_prin_comp):
    ###########################################################################
    # Extract information from data
    ###########################################################################
    N, D = data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D -= 1
    ###########################################################################
    # Separate each digit class data and extract useful information
    ###########################################################################
    sep_data = []
    sep_num_data = []
    sep_mean_data = []
    sep_within_cls_sctr = []
    n_digit = int(np.max(data[:, D]))
    num_class = n_digit+1
    for digit in range(num_class):
        class_data = data[data[:, D] == np.asarray(digit)]
        class_size = np.size(class_data, 0)
        class_mean = np.mean(class_data[:, 0:D], axis=0)
        class_diff_from_mean = class_data[:, 0:D] - class_mean
        within_class_scatter = class_diff_from_mean.T @ class_diff_from_mean
        sep_data.append(class_data)
        sep_num_data.append(class_size)
        sep_mean_data.append(class_mean)
        sep_within_cls_sctr.append(within_class_scatter)
    ###########################################################################
    # Calculate scatter of the means
    ###########################################################################
    sep_mean_data = np.asarray(sep_mean_data)
    overall_mean = np.mean(sep_mean_data, axis=0)
    ###########################################################################
    # Calculate total within class scatter and between class scatter
    ###########################################################################
    s_within = np.sum(sep_within_cls_sctr, axis=0)
    sep_num_data = np.asarray(sep_num_data)
    cls_diff_frm_ov_mean = (sep_mean_data-overall_mean)
    s_between = ((cls_diff_frm_ov_mean.T)*sep_num_data) @ cls_diff_frm_ov_mean
    ###########################################################################
    # Find projection matrix
    ###########################################################################
    w = (np.linalg.pinv(s_within)) @ s_between
    ###########################################################################
    # Compute eigen-values and eigen-vectors of projection matrix and sort it
    ###########################################################################
    eps = 1e-8
    e_val, e_vec = np.linalg.eig(w)
    e_val[np.abs(e_val) < eps] = 0
    e_val_abs = np.absolute(e_val)
    e_sort_index = np.argsort(e_val_abs)[::-1]
    e_val = np.real(e_val[e_sort_index])
    e_vec = np.real(e_vec[:, e_sort_index])
    return e_vec[:, 0:n_prin_comp], e_val[0:n_prin_comp]
