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
    # Plot each digit with different colors
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
# We need to plot first 5 eigen-faces for this question
num_eig_face = 5
w, e_val = mp.myPCA(all_data, num_eig_face)
###########################################################################
# Visualize eigen-faces
###########################################################################
n_image = 5
height = 30
width = 32
title = 'Mean face'
# For visualization purpose plot mean face
mu_all_data = np.mean(all_data[:, 0:D], axis=0)
plot_images(mu_all_data.reshape(1, D), 1, height, width, title)
title = 'First ' + str(n_image) + ' eigen-faces'
plot_images(w.T, n_image, height, width, title)
plt.show()
