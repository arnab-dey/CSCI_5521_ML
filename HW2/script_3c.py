###########################################################################
# Imports
###########################################################################
import numpy as np
import os, sys
import myPCA as mp
import myKNN as mk
import matplotlib.pyplot as plt
###########################################################################
# Function to plot images
###########################################################################
def plot_images(data, num_image, height, width, title):
    fig = plt.figure()
    ###########################################################################
    # Plot images in subplot
    ###########################################################################
    size = height*width
    for image in range(num_image):
        ax = fig.add_subplot(1, num_image, image+1)
        plt.imshow(np.reshape(data[image, 0:size], (height, width)))
    fig.suptitle(title)
    plt.draw()
###########################################################################
# Training, Test dataset location
###########################################################################
K = [10, 50, 100]
training_data = "./face_train_data_960.txt"
testing_data = "./face_test_data_960.txt"
###########################################################################
# Check if input files are present in the location
###########################################################################
if not os.path.isfile(training_data):
    sys.exit("Training data file can't be located")
if not os.path.isfile(testing_data):
    sys.exit("testing data file can't be located")
###########################################################################
# Load training data and extract information
###########################################################################
trn_data = np.loadtxt(training_data)
N, D = trn_data.shape
# Last column represents class label, therefore feature dimension is D-1
D -= 1
###########################################################################
# Load test data and extract information
###########################################################################
test_data = np.loadtxt(testing_data)
N_test, D_test = test_data.shape
# Last column represents class label, therefore feature dimension is D_test-1
D_test -= 1
###########################################################################
# PCA
###########################################################################
all_data = np.vstack((trn_data, test_data))
# We need to have max 100 components. Therefore we can choose any
# components greater than 100 to return. Then we can slice it
# accordingly
n_components = 150
w, e_val = mp.myPCA(all_data, n_components)
###########################################################################
# Visualize reconstructed image
###########################################################################
n_image = 5
height = 30
width = 32
# subtract mean before projection
mu_trn_data = np.mean(trn_data[:, 0:D], axis=0)
centered_trn = trn_data[:, 0:D] - mu_trn_data
# First visualize first 5 original images
title = 'First ' + str(n_image) + ' original images'
plot_images(trn_data, n_image, height, width, title)
for num_comp in K:
    # Apply projection
    reduced_trn_data = centered_trn[:, 0:D] @ w[:, 0:num_comp]
    D_all_reduced = reduced_trn_data.shape[1]
    # back projection
    recons_data = (reduced_trn_data @ w[:, 0:num_comp].T) + mu_trn_data
    title = 'First ' + str(n_image) + ' reconstructed images with K = ' + str(num_comp)
    plot_images(recons_data, n_image, height, width, title)
plt.show()
