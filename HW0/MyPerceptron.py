import numpy as np
import matplotlib.pyplot as plt


def MyPerceptron(X, y, w0):
    if (X is None) or (y is None) or (w0 is None):
        print("Invalid parameters")
        return
    ###########################################################################
    # Convert all parameters to numpy array to avoid error in case of
    # in case the inputs are not numpy arrays
    ###########################################################################
    x_train = np.asarray(X)
    y_label = np.asarray(y)
    w_init = np.asarray(w0)

    ###########################################################################
    # Get no. of samples and feature length
    ###########################################################################
    num_data = x_train.shape[0]
    num_feat = x_train.shape[1]
    if y_label.shape[0] < num_data:
        print("insufficient number of labels")
        return
    w_init = np.reshape(w_init, (1, num_feat))
    ###########################################################################
    # Sort data for plotting with different colors
    ###########################################################################
    c_1 = c_minus_1 = 0
    for sample in range(num_data):
        if y_label[sample] == 1:
            c_1 += 1
        else:
            c_minus_1 += 1
    x_1 = np.ones((c_1, num_feat))
    x_minus_1 = np.ones((c_minus_1, num_feat))
    index_1 = index_minus_1 = 0
    for sample in range(num_data):
        if y_label[sample] == 1:
            x_1[index_1] = x_train[sample]
            index_1 += 1
        else:
            x_minus_1[index_minus_1] = x_train[sample]
            index_minus_1 += 1
    x_sorted = (x_1, x_minus_1)

    ###########################################################################
    # Plot initialization figure
    ###########################################################################
    x_axis = np.array([max(x_train[:, 0]), min(x_train[:, 0])])
    y_axis = (-w_init[0, 0]/w_init[0, 1])*x_axis
    plt.figure(1)
    plt.title("Initialization")
    colors = ['red', 'blue']
    for data, clr in zip(x_sorted, colors):
        if clr == "red":
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="*")
        else:
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="P")
    plt.plot(x_axis, y_axis)
    x_lim = (plt.gca()).get_xlim()
    y_lim = (plt.gca()).get_ylim()
    plt.draw()
    #plt.savefig("init.png")
    ###########################################################################
    # Run Perceptron Algorithm
    ###########################################################################
    w = w_init
    err = True
    step = 0
    while err:
        step += 1
        for sample in range(num_data):
            if np.sign(x_train[sample]@w.T) != y_label[sample]:
                w = w + (y_label[sample]*x_train[sample])
        err = np.sum(np.sign(x_train@w.T) != y_label)
    ###########################################################################
    # Plot converged figure
    ###########################################################################
    y_axis = (-w[0, 0]/w[0, 1])*x_axis
    plt.figure(2)
    plt.title("Converged")
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    colors = ['red', 'blue']
    for data, clr in zip(x_sorted, colors):
        if clr == "red":
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="*")
        else:
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="P")
    plt.plot(x_axis, y_axis)
    #plt.savefig("converged.png")
    plt.show()
    plt.close(1)
    plt.close(2)
    return w[0, :], step
