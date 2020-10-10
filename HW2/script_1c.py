###########################################################################
# Imports
###########################################################################
import MultiGaussian as mg

###########################################################################
# Training, Test dataset location
###########################################################################
Model = [1, 2, 3]
train_data1 = "./training_data1.txt"
test_data1 = "./test_data1.txt"
train_data2 = "./training_data2.txt"
test_data2 = "./test_data2.txt"
train_data3 = "./training_data3.txt"
test_data3 = "./test_data3.txt"
train_data_arr = [train_data1, train_data2, train_data3]
test_data_arr = [test_data1, test_data2, test_data3]

# Learn parameters and get best prior from training and validation data set
for index in range(len(train_data_arr)):
    for model in Model:
        mg.MultiGaussian(train_data_arr[index], test_data_arr[index], model)
