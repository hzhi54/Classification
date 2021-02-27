import numpy as np
from sklearn.preprocessing import normalize
import math

def readFile(filename):
    contents = open(filename, 'r')
    firstLine = contents.readline().split()
    contents.close()
    num = [i for i in range(0, len(firstLine))]

    features = np.genfromtxt(filename, usecols=num, dtype='str')

    return features

def n_folds_cross_validation(n_samples, n_folds):
    number_each_fold = np.round(n_samples/n_folds).astype(int)

    retArr = []
    index = 0
    for i in range(n_folds):
        if(i!=n_folds - 1):
            retArr.append([x for x in range(index,(index+number_each_fold))])
            index += number_each_fold
        else:
            retArr.append([x for x in range(index,n_samples)])
    return retArr

def pdf(mean,sd,value):
    result = (1/ math.sqrt(2*math.pi * sd**2)) * math.exp(- (value - mean)**2 / (2 * sd**2))
    return result

def bayes(training_set,testing_set):
    TP, FN, FP, TN= 0, 0, 0, 0

    rows, cols = training_set.shape

    post = {}
    post[0] = np.count_nonzero(training_set[:, -1] == '0') / rows  # ph
    post[1] = 1 - post[0]

    for test_sample in testing_set:
        test_sample_zero ,test_sample_one, prior = [], [] , {0:post[0] , 1:post[1]}

        for row in training_set : test_sample_zero.append(row) if row[-1] == '0' else test_sample_one.append(row)   # Sepeate the training set into two subtree which belongs to two different classes

        test_sample_zero,test_sample_one = np.asarray(test_sample_zero), np.asarray(test_sample_one)

        for index,feature in enumerate(test_sample[:-1]):

            if feature.replace(".","").isdigit():                       # If feature is continues to calculate the probability by pdf
                zero_col , one_col = test_sample_zero[:,index].astype(float) , test_sample_one[:,index].astype(float)
                prior[0] , prior[1] = prior[0] * pdf(np.mean(zero_col),np.std(zero_col),float(feature)) ,  prior[1] * pdf(np.mean(one_col),np.std(one_col),float(feature))

            else:                                                       # If feature is nominals to calculate the probability
                nominal_zero, nominal_one = np.count_nonzero(test_sample_zero[:,index] == feature) , np.count_nonzero(test_sample_one[:,index] == feature)
                prior[0], prior[1] = prior[0] * (nominal_zero / len(test_sample_zero)) , prior[1] * (nominal_one / len(test_sample_one))

        label =  max(prior, key=lambda k: prior[k])                     # To determine the result label

        if int(test_sample[-1]) == 1 and int(label) == 1:
            TP += 1
        elif int(test_sample[-1]) == 1 and int(label) == 0:
            FN += 1
        elif int(test_sample[-1]) == 0 and int(label) == 1:
            FP += 1
        elif int(test_sample[-1]) == 0 and int(label) == 0:
            TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F_measure = 2 * recall * precision / (recall + precision)

    return accuracy, precision , recall , F_measure

def running(features, folds_arr):
    all_fold = [x for x in range(features.shape[0])]
    result = []
    for test_fold in folds_arr:
        train_fold = np.setdiff1d(all_fold,test_fold)

        testing_set, training_set = np.asarray([features[i] for i in test_fold]), np.asarray([features[i] for i in train_fold])

        result.append(bayes(training_set,testing_set))

    result = np.asarray(result)

    print("Accuracy : " + np.str(np.mean(result[:,0])))
    print("Precision : " + np.str(np.mean(result[:,1])))
    print("Recall : " + np.str(np.mean(result[:,2])))
    print("F_measure : " + np.str(np.mean(result[:,3])))


if __name__ == "__main__":

    filename = 'project3_dataset2.txt'

    features = readFile(filename)

    rows, cols = features.shape

    folds_arr = n_folds_cross_validation(rows,10)

    running(features, folds_arr)