###########################################################################
# Imports
###########################################################################
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import os
###########################################################################
# Function Definitions
###########################################################################
def EM_E_step(data, m_i, s_i, pi_i):
    h_i = []
    posterior_per_cluster = []
    N_data = (data.shape)[0]
    posterior_denom = np.zeros((data.shape[0]))
    N_cluster = (m_i.shape)[0]
    try:
        for cluster_index in range(N_cluster):
            membership_probability = multivariate_normal.pdf(data, m_i[cluster_index],
                                                             s_i[cluster_index])
            posterior_num = pi_i[cluster_index]*membership_probability
            posterior_per_cluster.append(posterior_num)
            posterior_denom += posterior_num
    except:
        print("Here sigma becomes singular. Therefore, its inverse does not exists. Hence, we cant calculate posterior probability."
              "returning None")
        return None
    for cluster_index in range(N_cluster):
        h_i.append(posterior_per_cluster[cluster_index]/posterior_denom)
    h_i = (np.vstack(h_i)).T
    return h_i

def EM_M_step(data, h_i, cluster_size, N_features, flag):
    N_data = data.shape[0]
    pi_i_next = []
    m_i_next = []
    sigma_i_next = []
    lamb = 0.00001
    for cluster_index in range(cluster_size):
        h_i_itr = h_i[:, cluster_index]
        sum_h_i = np.sum(h_i_itr, axis=0)
        pi_i_next.append(sum_h_i/N_data)
        sum_h_i_data = np.sum((data.T*h_i_itr).T, axis=0)
        m_i_next_itr = sum_h_i_data/sum_h_i
        m_i_next.append(m_i_next_itr)
        diff_from_updated_mean = data - m_i_next_itr
        if flag == 1:
            sigma_num = (((diff_from_updated_mean).T*h_i_itr).T).T @ diff_from_updated_mean + lamb*np.identity(N_features)
        else:
            sigma_num = (((diff_from_updated_mean).T * h_i_itr).T).T @ diff_from_updated_mean
        sigma_next_itr = sigma_num/sum_h_i
        sigma_i_next.append(sigma_next_itr)

    m_i_next = np.asarray(m_i_next)
    pi_i_next = np.asarray(pi_i_next)
    return pi_i_next, m_i_next, sigma_i_next

def getCompleteLogLikelihood(data, h_i, m_i, sigma_i, pi_i, k):
    prob_x_given_phi = []
    try:
        for cluster_index in range(k):
            membership_probability = multivariate_normal.pdf(data, m_i[cluster_index],
                                                         sigma_i[cluster_index])
            prob_x_given_phi.append(membership_probability)
    except:
        return None
    prob_x_given_phi = (np.vstack(prob_x_given_phi)).T
    sum_1 = np.log(pi_i) + np.log(prob_x_given_phi+1e-8)
    sum_2 = sum_1.T @ h_i
    expected_complete_likehihood = np.sum(np.diag(sum_2), axis=0)
    return expected_complete_likehihood


def EMG(imagepath, k, flag):
    ###########################################################################
    # Check if input files are present in the location
    ###########################################################################
    if not os.path.isfile(imagepath):
        print("image file can't be located")
        return None, None, None, None, None, None
    img = io.imread(imagepath)
    img = img / 255
    ###########################################################################
    # Reshaping image to represent RGB values column wise
    ###########################################################################
    height,width,col_depth = img.shape
    img = np.reshape(img, ((height*width), col_depth))
    N = (img.shape)[0] # number of samples
    D = (img.shape)[1] # feature dimension
    ###########################################################################
    # Initialize cluster using K-means to start EM
    ###########################################################################
    kmeans = KMeans(n_clusters = k, n_init = 1, max_iter = 3).fit(img)
    ###########################################################################
    # Estimate initial values of centers
    ###########################################################################
    m_i = kmeans.cluster_centers_
    ###########################################################################
    # Estimate initial values of posteriors
    ###########################################################################
    ###########################################################################
    # Estimate variance sigma from k-means
    ###########################################################################
    posterior = kmeans.labels_
    prior_arr = []
    sigma_i = []
    for num_cluster in range(k):
        bin_data = img[posterior[:] == num_cluster]
        N_bin = (bin_data.shape)[0]
        cluster_posterior = N_bin/N
        prior_arr.append(cluster_posterior)
        # Estimate sigma
        diff_from_mean = bin_data - m_i[num_cluster]
        sigma = diff_from_mean.T @ diff_from_mean / N_bin
        sigma_i.append(sigma)
    pi_i = np.asarray(prior_arr)
    ###########################################################################
    # Run EM
    ###########################################################################
    num_iter_em = 200
    exp_log_likelihood_arr = []
    error_flag = False
    for iteration in range(num_iter_em):
        ###########################################################################
        # E-step
        ###########################################################################
        h_i = EM_E_step(img, m_i, sigma_i, pi_i)
        if h_i is None:
            error_flag = True
            break
        ###########################################################################
        # M-step
        ###########################################################################
        pi_i, m_i, sigma_i = EM_M_step(img, h_i, k, D, flag)
        exp_log_likelihood = getCompleteLogLikelihood(img, h_i, m_i, sigma_i, pi_i, k)
        exp_log_likelihood_arr.append(exp_log_likelihood)
    exp_log_likelihood_arr = np.asarray(exp_log_likelihood_arr)
    ###########################################################################
    # return cluster posteriors and mean
    ###########################################################################
    if error_flag is True:
        error_flag = False
        return None,None,None
    ###########################################################################
    # Reconstruct image and show
    ###########################################################################
    compressed = np.argmax(h_i, axis=1)
    compressed_image = []
    for pixel in range(N):
        color = compressed[pixel]
        compressed_image.append(m_i[color, :])
    ###########################################################################
    # Reconstruct image and show
    ###########################################################################
    compressed_image = np.reshape(compressed_image, (height, width, col_depth))
    io.imshow(compressed_image)
    io.show()
    return h_i, m_i, exp_log_likelihood_arr



    


