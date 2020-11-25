"""

"""

import numpy as np
import torch
import utils
from tqdm import tqdm
import multiprocessing
import torch.nn as nn
from parse import args
import time


def BPR_train_original(dataset, Recmodel, bpr: utils.BPRLoss, epoch, w=None, val=-1):
    if not args.simutaneously:
        # train alternatively
        Recmodel.train()
        for m in Recmodel.modules():
            if isinstance(m, nn.Embedding):
                m.requires_grad_(val <= 0)
            elif hasattr(m, 'weight'):
                m.requires_grad_(not (val <= 0))
        print('mapping fixed') if val <= 0 else print('embedding fixed')
    # for m in Recmodel.modules():  # test separate training
    #     if hasattr(m, 'weight'):
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             print(m, 'weight:', m.weight.requires_grad, ' bias:', m.bias.requires_grad)
    #         else:
    #             print(m, 'weight', m.weight.requires_grad)

    S, sam_time = utils.UniformSample_original(dataset, val)  # bpr sample
    print(f"sample time:{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}")
    users = torch.Tensor(S[:, 0]).long().to(args.device)
    posItems = torch.Tensor(S[:, 1]).long().to(args.device)
    negItems = torch.Tensor(S[:, 2]).long().to(args.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // args.bpr_batch_size + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) \
            in enumerate(utils.minibatch(users, posItems, negItems, batch_size=args.bpr_batch_size)):
        # train on different graph
        cri = bpr.stageOne(dataset.graph[val+1], batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if args.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / args.bpr_batch_size) + batch_i)
    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def test(dataset, Recmodel, epoch, w=None, multicore=0, best_results=None):
    u_batch_size = args.test_batch
    testDict = dataset.testDict
    Recmodel = Recmodel.eval()  # eval mode with no dropout
    max_K = max(args.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(args.n_cores)
    results = {'precision': np.zeros(len(args.topks)),
               'recall': np.zeros(len(args.topks)),
               'ndcg': np.zeros(len(args.topks))}

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list, rating_list, groundTrue_list, auc_record = [], [], [], []
        t0 = time.time()
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users, dataset.UserItemNet)  # **********
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)

            rating = Recmodel.getUsersRating(dataset.graph[args.test_graph], batch_users_gpu)  # **********
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            aucs = [utils.AUC(rating[i], dataset, test_data) for i, test_data in enumerate(groundTrue)]
            auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['auc'] = np.mean(auc_record)
        if args.tensorboard:
            w.add_scalars(f'Test/Recall@{args.topks}',
                          {str(args.topks[i]): results['recall'][i] for i in range(len(args.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{args.topks}',
                          {str(args.topks[i]): results['precision'][i] for i in range(len(args.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{args.topks}',
                          {str(args.topks[i]): results['ndcg'][i] for i in range(len(args.topks))}, epoch)
        if multicore == 1:
            pool.close()

        print('time consumption: %.2f' % (time.time()-t0))
        print('recall:', results['recall'], 'precision:', results['precision'],
              'ndcg:', results['ndcg'], 'auc:', results['auc'])

        if results['recall'][0] > best_results['recall'][0]:
            return results
        else:
            return best_results
