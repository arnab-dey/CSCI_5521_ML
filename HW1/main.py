###########################################################################
# Imports
###########################################################################
import Bayes_Learning as bL
import Bayes_testing as bT

###########################################################################
# Training, Cross-validation and Test dataset location
###########################################################################
train_data = "./training_data.txt"
val_data = "./validation_data.txt"
test_data = "./testing_data.txt"

# Learn parameters and get best prior from training and validation data set
p1, p2, p_c1, p_c2 = bL.Bayes_Learning(train_data, val_data)

# Classify samples of test dataset
if p1 is not None or p2 is not None or p_c1 is not None or p_c2 is not None:
    bT.Bayes_Testing(test_data, p1, p2, p_c1, p_c2)