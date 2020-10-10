###########################################################################
# Imports
###########################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import kernPercGD as kp
from sklearn import svm
###########################################################################
# Function definitions
###########################################################################
def predict(test, train, pol_degree, alpha, b):
    labels = train[:, -1]
    x_train = train[:, 0:-1]
    K = kp.getGramMat(test, x_train, pol_degree)
    score = ((alpha.T * labels) @ K).T+b
    prediction = np.sign(score)
    return prediction, score

def plot_contour(data, pol_degree, sv, alpha, b, clf=None):
    # Visualize data
    plt.figure()
    labels = data[:, -1]
    plt.scatter(data[np.where(labels == 1)[0], 0],
                data[np.where(labels == 1)[0], 1], c='r')
    plt.scatter(data[np.where(labels == -1)[0], 0],
                data[np.where(labels == -1)[0], 1], c='b')
    # Plot contour
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    test_data = np.vstack([xx.ravel(), yy.ravel()]).T
    predicted, score = predict(test_data, sv, pol_degree, alpha, b)
    score = score.reshape(xx.shape)
    cp = plt.contour(xx, yy, score, [0.0], colors='k', linewidths=1, origin='lower')
    plt.axis('tight')
    cp.collections[0].set_label('my contour')
    if clf is not None:
        z = clf.decision_function(test_data)
        z = z.reshape(xx.shape)
        cp_skl = plt.contour(xx, yy, z, [0.0], colors='g', linewidths=1, origin='lower')
        cp_skl.collections[0].set_label('sklearn contour')
    plt.legend()
    plt.show()

def create_dummy_data():
    # Create random data
    np.random.seed(1)  # For reproducibility
    r1 = np.sqrt(np.random.rand(100, 1))  # Radius
    t1 = 2 * np.pi * np.random.rand(100, 1)  # Angle
    data1 = np.hstack((r1 * np.cos(t1), r1 * np.sin(t1)))  # Points
    np.random.seed(2)  # For reproducibility
    r2 = np.sqrt(3 * np.random.rand(100, 1) + 2)  # Radius
    t2 = 2 * np.pi * np.random.rand(100, 1)  # Angle
    data2 = np.hstack((r2 * np.cos(t2), r2 * np.sin(t2)))  # Points

    # Combine all data and add labels
    data3 = np.vstack((data1, data2))
    labels = np.ones((200, 1))
    labels[0:100, :] = -1

    data = np.hstack((data3, labels))
    np.random.shuffle(data)
    return data

###########################################################################
# Main function
###########################################################################
data = create_dummy_data()
labels = data[:, -1]

# plt.show()
poly_degree = 3
alpha, b = kp.kernPercGD(data[:, 0:-1], data[:, -1], poly_degree)
sv = alpha > 1e-5
alpha = alpha[sv]
sv = data[sv[:, 0], :]
sv_y = sv[:, -1]
# Plot data with contour
plot_contour(data, poly_degree, sv, alpha, b)

###########################################################################
# SVM using sklearn
###########################################################################
X = data[:, 0:-1]
Y = data[:, -1]
clf = svm.SVC(kernel=kp.polynomial_kernel, degree=poly_degree, C=0.05)
clf.fit(X, Y)
plot_contour(data, poly_degree, sv, alpha, b, clf)
###########################################################################
# Effect of C parameter
###########################################################################
C = [0.05, 0.005, 0.0005]
color = ['m', 'c', 'k']
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
test_data = np.vstack([xx.ravel(), yy.ravel()]).T
plt.figure()
plt.scatter(X[np.where(Y == 1)[0], 0],
                X[np.where(Y == 1)[0], 1], c='r')
plt.scatter(X[np.where(Y == -1)[0], 0],
            X[np.where(Y == -1)[0], 1], c='b')
for index in range(len(C)):
    clf = svm.SVC(kernel=kp.polynomial_kernel, degree=poly_degree, C=C[index])
    clf.fit(X, Y)
    z = clf.decision_function(test_data)
    z = z.reshape(xx.shape)
    cp_skl = plt.contour(xx, yy, z, [0.0], colors=color[index], linewidths=1, origin='lower')
    label = 'C = ' + str(C[index])
    cp_skl.collections[0].set_label(label)
plt.legend()
plt.show()
###########################################################################
# Q1.c: Train on optdigits 4 and 9
###########################################################################
training_filename = './optdigits49_train.txt'
test_filename = './optdigits49_test.txt'
training_data = np.loadtxt(training_filename, delimiter=',')
test_data = np.loadtxt(test_filename, delimiter=',')
# Learn using polynomial kernel
alpha, b = kp.kernPercGD(training_data[:, 0:-1], training_data[:, -1], poly_degree)
sv = alpha > 1e-5
alpha = alpha[sv]
sv = training_data[sv[:, 0], :]
sv_y = sv[:, -1]
# Predict on test data
prediction, score = predict(test_data[:, 0:-1], sv, poly_degree, alpha, b)
err = (np.sum(prediction != test_data[:, -1]))*100./test_data.shape[0]
print('Error on ', test_filename, ' is ', err, '%')
###########################################################################
# Q1.c: Train on optdigits 7 and 9
###########################################################################
training_filename = './optdigits79_train.txt'
test_filename = './optdigits79_test.txt'
training_data = np.loadtxt(training_filename, delimiter=',')
test_data = np.loadtxt(test_filename, delimiter=',')
# Learn using polynomial kernel
alpha, b = kp.kernPercGD(training_data[:, 0:-1], training_data[:, -1], poly_degree)
sv = alpha > 1e-5
alpha = alpha[sv]
sv = training_data[sv[:, 0], :]
sv_y = sv[:, -1]
# Predict on test data
prediction, score = predict(test_data[:, 0:-1], sv, poly_degree, alpha, b)
err = (np.sum(prediction != test_data[:, -1]))*100./test_data.shape[0]
print('Error on ', test_filename, ' is ', err, '%')