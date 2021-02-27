import numpy as np
from sklearn.preprocessing import normalize
import random
import sys
from collections import Counter

def readFile(filename):
    contents = open(filename, 'r')
    firstLine = contents.readline().split()
    contents.close()
    num = [i for i in range(0, len(firstLine) - 1)]

    features = np.genfromtxt(filename, usecols=num, dtype='str')

    labels = np.genfromtxt(filename, usecols= len(firstLine) -1 , dtype= 'str')

    return features, labels

def pre_processing(data):
    cols , rows = data.shape

    nominal_data = np.asarray([data[:,i] for i in range(rows) if not data[0,i].replace(".","").isdigit()])

    continue_data = np.asarray([data[:,i] for i in range(rows) if data[0,i].replace(".","").isdigit()])

    return nominal_data.T, continue_data.T

def data_normalization(features):
    return normalize(features, axis=0, norm='l1')

def n_folds_cross_validation(n_samples, n_folds, labels):
    number_each_fold = np.round(n_samples/10).astype(int)

    retArr = []
    index = 0
    for i in range(n_folds):
        if(i!=9):
            retArr.append([x for x in range(index,(index+number_each_fold))])
            index += number_each_fold
        else:
            retArr.append([x for x in range(index,len(labels))])
    return retArr

def vote(labels,index_arr,k_nn):
    k_nn_index = [index_arr[i] for i in k_nn]
    k_nn_labels = [labels[i] for i in k_nn_index]
    c = Counter(k_nn_labels)
    return int(c.most_common(1)[0][0])

def classification(nominal_features, continue_featuers, folds_arr , n_samples, labels, k):
    TP, FN, FP, TN = 0, 0, 0, 0
    a, p, r, f = [], [], [], []

    all_index = [x for x in range(n_samples)]

    for test_index in folds_arr:
        training_index = np.setdiff1d(all_index, test_index)

        training_continue = [continue_featuers[i] for i in training_index]
        training_nominal = []
        if (len(nominal_features) != 0):
            training_nominal = [nominal_features[i] for i in training_index]

        for index in test_index:
            sample = continue_featuers[index]
            training_data = training_continue
            nominal_sample = []
            if(len(nominal_features) != 0):
                curr_nominal = nominal_features[index]
                sample = np.append(sample,[0 for i in range(len(nominal_features[0]))])

                train_nominal_compare_sample = [[0 if arr[i] != curr_nominal[i] else 1 for i in range(len(arr)) ] for arr in training_nominal]

                training_data = np.append(training_continue,train_nominal_compare_sample,axis=1)

            result = [np.linalg.norm(i - sample) for i in training_data]
            k_nn = np.argsort(result)[:k]
            label = vote(labels, training_index, k_nn)

            if int(labels[index]) == 1 and label == 1:
                    TP += 1
            elif int(labels[index]) == 1 and label == 0:
                    FN += 1
            elif int(labels[index]) == 0 and label == 1:
                    FP += 1
            elif int(labels[index]) == 0 and label == 0:
                    TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = 2 * recall * precision / (recall + precision)

        a.append(accuracy)
        p.append(precision)
        r.append(recall)
        f.append(F_measure)

    print("Accuracy : " + np.str(np.mean(a)))
    print("Precision : " + np.str(np.mean(p)))
    print("Recall : " + np.str(np.mean(r)))
    print("F_measure : " + np.str(np.mean(f)))

if __name__ == "__main__":
    filename = 'project3_dataset1.txt'

    features, labels = readFile(filename)

    nominal_features , continue_features = pre_processing(features)

    normalized_continue_features = data_normalization(continue_features)

    n_samples, n_features = features.shape

    folds_arr = n_folds_cross_validation(n_samples, 10)           # First argument is number of samples, second is number of folds.

    k = 10

    classification(nominal_features, normalized_continue_features, folds_arr, n_samples , labels, k)