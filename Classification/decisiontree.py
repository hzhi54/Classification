import numpy as np
import sys
from sklearn.preprocessing import normalize


def readFile(filename):
    contents = open(filename, 'r')
    firstLine = contents.readline().split()
    contents.close()
    num = [i for i in range(0, len(firstLine)-1)]

    features = np.genfromtxt(filename, usecols=num, dtype='str')

    labels = np.genfromtxt(filename, usecols=len(firstLine)-1, dtype='str')

    return features, labels


def pre_processing(data):
    cols, rows = data.shape

    nominal_data = np.asarray([data[:, i] for i in range(rows) if not data[0, i].replace(".", "").isdigit()])

    continue_data = np.asarray([data[:, i] for i in range(rows) if data[0, i].replace(".", "").isdigit()])

    return nominal_data.T, continue_data.T


def data_normalization(features):
    return normalize(features, axis=0, norm='l1')


def n_folds_cross_validation(n_samples, n_folds, labels):
    number_each_fold = np.round(n_samples/n_folds).astype(int)

    retArr = []
    index = 0
    for i in range(n_folds):
        if i != 9:
            retArr.append([x for x in range(index, (index+number_each_fold))])
            index += number_each_fold
        else:
            retArr.append([x for x in range(index, len(labels))])
    return retArr


def find_split(dataset, value, index):
    left = []
    right = []

    for d in dataset:
        if np.float(d[index]) < value:
            left.append(d)
        else:
            right.append(d)

    return np.array(left), np.array(right)


def gini_index(left, right):
    labels = ['0','1']
    l_size = float(len(left))
    r_size = float(len(right))
    sample_size = (l_size + r_size)
    gini_value = 0
    if len(left) > 0:
        score = 0.0
        for l in labels:
            pr = list(left[:,-1]).count(l) / l_size
            score += (pr*pr)
        gini_value += (1.0 - score) * (l_size / sample_size)

    if len(right) > 0:
        score = 0.0
        for l in labels:
            pr = list(right[:,-1]).count(l)/r_size
            score += (pr*pr)
        gini_value += (1.0 - score) * (r_size / sample_size)

    return gini_value


def best_split(dataset):
    # unique_label = np.unique(sample[-1] for sample in dataset)
    split = None
    _score = sys.maxsize
    for idx in range(dataset.shape[1]-1):
        for d in dataset:
            left, right = find_split(dataset, np.float(d[idx]), idx)
            gini_val = gini_index(left, right)
            if gini_val < _score:
                _score = gini_val
                split = {'index': idx, 'value': d[idx], 'left': left, 'right': right, 'gini': _score}
                # print(split)
    return split


def terminal_node(left, right):
    count_z = 0
    count_o = 0

    if len(left) > 0:
        count_z += list(left[:,-1]).count('0')
        count_o += list(left[:,-1]).count('1')
    if len(right) > 0:
        count_z += list(right[:,-1]).count('0')
        count_o += list(right[:,-1]).count('1')

    if count_z > count_o:
        return 0
    else:
        return 1


def child_split(node):
    left = node['left']
    right = node['right']
    node.pop('left', None)
    node.pop('right', None)

    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = terminal_node(left, right)
        return node

    if len(set(left[:,-1])) == 1:
        node['left'] = terminal_node(left, [])
    else:
        node['left'] = child_split(best_split(left))

    if len(set(right[:,-1])) == 1:
        node['right'] = terminal_node([], right)
    else:
        node['right'] = child_split(best_split(right))
    return node


def build_tree(train):
    root = best_split(train)
    child_split(root)
    return root


def predict(node, samples):
    if np.float(samples[node['index']]) < np.float(node['value']):
        if isinstance(node['left'], dict):
            return predict(node['left'], samples)
        else:
            return node['left']
    if isinstance(node['right'], dict):
        return predict(node['right'], samples)
    else:
        return node['right']


def decision_tree(train, test):
    tree = build_tree(train)
    predictions = []
    for samples in test:
        prediction = predict(tree, samples)
        predictions.append(prediction)
    return predictions


def measure_metrics(guess, labels):
    TP, FN, FP, TN = 0, 0, 0, 0

    for idx in range(len(labels)):
        if np.int(labels[idx]) == 1 and (guess[idx]) == 1:
            TP += 1
        elif np.int(labels[idx]) == 1 and (guess[idx]) == 0:
            FN += 1
        elif np.int(labels[idx]) == 0 and (guess[idx]) == 1:
            FP += 1
        elif np.int(labels[idx]) == 0 and (guess[idx]) == 0:
            TN += 1

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F_measure = 2*recall*precision/(recall+precision)
    # print(accuracy)
    return [accuracy, precision, recall, F_measure]


def classify(dataset, features, num_fold, labels):
    n_samples, n_features = features.shape
    fold_arr = n_folds_cross_validation(n_samples, num_fold, labels)
    # chunks = np.array_split(dataset,num_fold)

    counter = 0
    chunks = []
    for fold in fold_arr:
        chunks.append(dataset[counter:counter+len(fold)])
        counter += len(fold)
    guess = []

    all_predicts = []
    all_labels = []

    print("Building Tree...")
    for idx in range(num_fold):
        predict_list = []
        training_set = np.array(np.concatenate([y for (x,y) in enumerate(chunks,0) if x != idx],axis=0))
        testing_set = np.array(chunks[idx])
        print(idx+1,"Finding Split")
        starting_root = best_split(training_set)
        root = child_split(starting_root)
        for td in testing_set:
            predict_list.append(predict(root,td))
        all_predicts.append(predict_list)
        all_labels.append(testing_set[:,-1])

    # print(all_predicts)
    # print(all_labels)


    results = [0,0,0,0]
    for idx in range(len(all_predicts)):
        # print(all_predicts[idx])
        # print(all_labels[idx])
        results[0] += measure_metrics(all_predicts[idx], all_labels[idx])[0]
        results[1] += measure_metrics(all_predicts[idx], all_labels[idx])[1]
        results[2] += measure_metrics(all_predicts[idx], all_labels[idx])[2]
        results[3] += measure_metrics(all_predicts[idx], all_labels[idx])[3]
    results[0] = results[0]/num_fold
    results[1] = results[1]/num_fold
    results[2] = results[2]/num_fold
    results[3] = results[3]/num_fold
    return results

def find_string_idx(dataset):
    idx = 0
    idx_map = []
    for d in dataset[0]:
        if d.isalpha():
            idx_map.append(idx)
        idx = idx + 1

    return idx_map


def convert_name_to_int(idx_map):
    name_map = {}
    for idx in idx_map:
        counter = 0
        for unique in temp[:,idx]:
            if unique not in name_map:
                name_map[unique] = str(counter)
                counter = counter + 1
    return name_map


if __name__ == "__main__":
    filename = "project3_dataset2.txt"
    # filename = "project3_dataset1.txt"
    features, labels = readFile(filename)
    # nominal_features , continue_features = pre_processing(features)
    # normalized_continue_features = data_normalization(continue_features)

    c = 0
    dataset = []
    for f in features:
        dataset.append(np.append(f, labels[c]).tolist())
        c += 1

    temp = np.asarray(dataset)
    idx_map = find_string_idx(dataset)
    # print(idx_map)
    if len(idx_map) > 0:
        name_map = convert_name_to_int(idx_map)
        for name in temp:
            for idx in idx_map:
                if name[idx] in name_map.keys():
                    name[idx] = name_map[name[idx]]

        dataset = temp.tolist()
        # print(name_map)
        # print(idx_name)

    # print(dataset)

    n = 10
    # n = 3
    result = classify(dataset, features, n, labels)
    print("Accuracy: ", result[0])
    print("Precision: ", result[1])
    print("Recall: ", result[2])
    print("F_measure: ", result[3])
