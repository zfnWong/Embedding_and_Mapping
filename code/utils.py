"""

"""

import torch
from torch import nn, optim
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
import random
import os
from parse import args


# ====================Loss==============================
# =========================================================
class BPRLoss:
    def __init__(self, recmodel):
        self.model = recmodel
        self.reg_lambda = args.reg_lambda
        self.lr = args.lr
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, graph, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(graph, users, pos, neg)
        reg_loss = reg_loss * self.reg_lambda
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


# ====================Sample==============================
# =========================================================
def UniformSample_original(dataset, val=-1):
    """
    the original impliment of BPR Sampling in LightGCN
    the num of samples is equal to the num of interactions
    Each sample is a random user
    output:
        S - np.array([n_interactions, 3])
        time - [total, sample_time1, sample_time2]
    """
    total_start = time()
    # len(users) is equal to n_interactions
    # the target of sample here is user
    users, pos = None, None
    if val == 0:
        users = np.random.choice(dataset.subtrainUniqueUsers, dataset.subtraindataSize)
        pos = dataset.trainPos
    elif val == 1:
        users = np.random.choice(dataset.valUniqueUsers, dataset.valdataSize)
        pos = dataset.valPos
    elif val == -1:
        users = np.random.choice(dataset.trainUniqueUsers, dataset.traindataSize)
        pos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = pos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        mid = time()
        sample_time1 += mid - start
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in dataset.allPos[user]:
                continue
            else:
                break
        S.append([user, positem, negitem])
        sample_time2 += time() - mid
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]


# ====================Utils==============================
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    file = None
    if args.model == 'mf':
        file = f"mf-{args.dataset}-{args.embed_dim}.pth.tar"
    else:
        file = f"lgn-{args.dataset}-{args.n_layers}-{args.embed_dim}.pth.tar"
    return os.path.join(args.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', args.bpr_batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


