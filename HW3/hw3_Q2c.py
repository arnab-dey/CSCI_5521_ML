###########################################################################
# Imports
###########################################################################
import numpy as np
from skimage import io
import EMG
from sklearn.cluster import KMeans
###########################################################################
# Run EM
###########################################################################
imagepath = './goldy.jpg'
k = 7
flag = 0
h, m, complete_likelihood = EMG.EMG(imagepath, k, flag)
if h is None or m is None or complete_likelihood is None:
    print('Proceeding with k-means')
    ###########################################################################
    # Proceed with k-means
    ###########################################################################
    img = io.imread(imagepath)
    img = img / 255
    ###########################################################################
    # Reshaping image to represent RGB values column wise
    ###########################################################################
    height, width, col_depth = img.shape
    img = np.reshape(img, ((height * width), col_depth))
    N = (img.shape)[0]  # number of samples
    D = (img.shape)[1]  # feature dimension
    ###########################################################################
    # Initialize cluster using K-means to start EM
    ###########################################################################
    kmeans = KMeans(n_clusters=k).fit(img)
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
    k_means_labels = kmeans.labels_
    ###########################################################################
    # Reconstruct image and show
    ###########################################################################
    compressed_image = []
    for pixel in range(N):
        color = k_means_labels[pixel]
        compressed_image.append(m_i[color, :])
    ###########################################################################
    # Reconstruct image and show
    ###########################################################################
    compressed_image = np.reshape(compressed_image, (height, width, col_depth))
    io.imshow(compressed_image)
    io.show()

