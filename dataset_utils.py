import numpy as np
def dataset_one_against_others(dataset, target_cls, train = True):
    '''
    Change the input dataset so that every non-target class has a label of 0, and target cls has a label of 1
    :param dataset: input dateset
    :param target_cls: Target class we want to classify out of all n classes
    :param train:  Change train labels or test labels
    :return: revised dataset
    '''
    if train:
        data = dataset.train_labels
    else:
        data = dataset.test_labels
    include_idx =  data == target_cls
    mask = np.zeros(dataset.data.shape[0], dtype=bool)
    mask[include_idx] = True
    data[mask] = 1
    data[~mask] = 0
    return dataset

def dataset_two_cls(dataset, cls1, cls2, train = True):
    '''
    Change the input multi-class dataset to 2-class dataset
    :param dataset: multi-class dataset
    :param cls1: target label1
    :param cls2: target label2
    :param train: Change train labels or test labels
    :return: Revised dataset
    '''
    if train:
        data = dataset.train_data
        labels = dataset.train_labels
    else:
        data = dataset.test_data
        labels = dataset.test_labels
    cls1idx = labels == cls1
    labels[cls1idx] = 0
    cls2idx = labels == cls2
    labels[cls2idx] = 1
    mask = np.zeros(dataset.data.shape[0], dtype=bool)
    mask[cls1idx] = True
    mask[cls2idx] = True
    labels = labels[mask]
    data = data[mask]
    return dataset

def drop_params(params, thres = 1e-3):
    '''
    Drop params that falles under a threshold
    :param params:all parameters in the model
    :param thres:dropout thres
    :return:freezed # if params
    '''
    cnt = 0
    for p in params:
        if not p.requires_grad:
            cnt+=1
            continue
        if p.data < thres:
            p.requires_grad = False
    return cnt



