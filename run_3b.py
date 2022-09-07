#!/usr/bin/env python
'''
This script contains various functions used in this project
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/13/2022
'''

import utilities
import random
import numpy as np
str_path_1b_program = './data_2c2d3c3d_program.txt'

#splitting data into train and test data sets
features,targets = utilities.Load_data(str_path_1b_program)

#changing label from M to1 and W to -1
targets_tmp = np.zeros([len(targets),1])
for i in range(len(targets)):
    if(targets[i]=='M'):
        targets_tmp[i] = int(1)
    else:
        targets_tmp[i] = int(-1)
targets_tmp = targets_tmp.ravel()

arr3d_train_features = features[0:90]
arr3d_test_features = features[90:120]
arr1d_train_targets = targets_tmp[0:90]
arr1d_test_targets = targets_tmp[90:120]

random.seed(10)
np.random.seed(10)
trainfeatures,trainlabels,testfeature,testlabels = arr3d_train_features,arr1d_train_targets,arr3d_test_features,arr1d_test_targets
bagging_times = [10,25,50]

for i in bagging_times:
    train_pred = utilities.adaboost(trainfeatures, trainlabels, 4, i, trainfeatures)
    test_pred = utilities.adaboost(trainfeatures, trainlabels, 4, i, testfeature)

    train_accuracy = utilities.accuracy(trainlabels, train_pred)
    test_accuracy = utilities.accuracy(testlabels, test_pred)
    print('Train Accuracy for boosting with ' +str(i)+' times ' +str(train_accuracy))
    print('Test Accuracy for boosting with ' +str(i)+' times ' +str(test_accuracy))
    print('\n')