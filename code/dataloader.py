"""

"""
from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import random
from parse import args
import re


class Loader(Dataset):
    """

    """
    def __init__(self, path="../data/gowalla"):
        # train or test
        super().__init__()
        print(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                l = re.split('[ \t]', l.strip())
                if len(l) > 1:
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # sample val set from train set and delete the repetitive interactions
        self.valdataSize = int(args.p_val * self.traindataSize)
        valIndex = random.sample(range(0, self.traindataSize), self.valdataSize)
        self.valUser = self.trainUser[valIndex]
        self.valItem = self.trainItem[valIndex]
        self.valUniqueUsers = np.unique(self.valUser)

        self.subtraindataSize = self.traindataSize - self.valdataSize
        trainIndex = list(set(range(0, self.traindataSize)) - set(valIndex))
        self.subtrainUser = self.trainUser[trainIndex]
        self.subtrainItem = self.trainItem[trainIndex]
        self.subtrainUniqueUsers = np.unique(self.subtrainUser)

        with open(test_file) as f:
            for l in f.readlines():
                l = re.split('[ \t]', l.strip())
                if len(l) > 1:
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        print(f"{self.trainDataSize} interactions, {len(trainUniqueUsers)} users, {len(set(trainItem))} items for training")
        print(f"{self.testDataSize} interactions, {len(testUniqueUsers)} users, {len(set(testItem))} items for testing")
        print(f"train item list contains all the item in test list: {set(trainItem) >= set(testItem)}")
        print(f"{args.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), sparse adj
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        trainUserItemNet = csr_matrix((np.ones(len(self.subtrainUser)), (self.subtrainUser, self.subtrainItem)),
                                      shape=(self.n_user, self.m_item))
        valUserItemNet = csr_matrix((np.ones(len(self.valUser)), (self.valUser, self.valItem)),
                                    shape=(self.n_user, self.m_item))

        # sparse norm adj
        self.graph = [self.getSparseGraph(self.UserItemNet, 'whole'),
                      self.getSparseGraph(trainUserItemNet, 'train'),
                      self.getSparseGraph(valUserItemNet, 'val')]

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)
        self._trainPos = self.getUserPosItems(list(range(self.n_user)), trainUserItemNet)
        self._valPos = self.getUserPosItems(list(range(self.n_user)), valUserItemNet)
        self.__testDict = self.__build_test()
        print(f"{args.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def trainPos(self):
        return self._trainPos

    @property
    def valPos(self):
        return self._valPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, user2item, name):
        try:
            pre_adj_mat = sp.load_npz(self.path + f'/{name}_adj_mat.npz')
            print("loading adjacency matrix")
            norm_adj = pre_adj_mat
        except:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = user2item.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(self.path + f'/{name}_adj_mat.npz', norm_adj)

        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        graph = graph.coalesce().to(args.device)
        return graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users, user2item):
        posItems = []
        for user in users:
            posItems.append(user2item[user].nonzero()[1])
        return posItems
