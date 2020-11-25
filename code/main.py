import utils
import torch
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import numpy as np
from parse import args
import model
from pprint import pprint
import dataloader

pprint(vars(args))
utils.set_seed(args.seed)

# dataset
dataset = dataloader.Loader(path="../data/"+args.dataset)

# model
MODELS = {
    'mf': model.PureMF,
    'ngcf': model.NGCF,    'lgn': model.LightGCN,
    'lgn_ws': model.LightGCN_ws,
    'lgn_ecc': model.LightGCN_ecc,
}
Recmodel = MODELS[args.model](dataset).to(args.device)
print(Recmodel)
weight_file = utils.getFileName()
print(f"model will be save in {weight_file}")

# loss
bpr = utils.BPRLoss(Recmodel)

# result
best_result = {'recall': np.array([0.0]),
               'precision': np.array([0.0]),
               'ndcg': np.array([0.0]),
               'auc': np.array([0.0])}

# init tensorboard
w: SummaryWriter = SummaryWriter(
    join(args.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.model)) if args.tensorboard else None

try:
    val = -1  # val -1:whole_train_set  0:subtrain_set  1:validation_set
    # last_loss, aver_loss = 0., 0.
    for epoch in range(args.epochs + args.fine_tune_epochs):
        if not args.simutaneously:
            # training
            if epoch < args.epochs:
                # apply adaptive training
                # if not (epoch > 1 and aver_loss > last_loss * args.p_dist):
                #     val = epoch // args.interval % 2
                #     last_loss = aver_loss
                val = epoch // args.interval % 2

            # fine tune towards embeddings
            elif epoch == args.epochs:
                print('[phase 1]best result: {recall:', best_result['recall'], 'precision:', best_result['precision'],
                      'ndcg:', best_result['ndcg'], 'auc:', best_result['auc'], '}')
                val = -1
                if args.epochs > 0:
                    Recmodel.load_state_dict(torch.load(weight_file, map_location=args.device))

        print('======================')
        print(f'EPOCH[{epoch + 1}/{args.epochs + args.fine_tune_epochs}]')
        start = time.time()
        aver_loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, w=w, val=val)  # train func
        print(f"loss:{aver_loss:.3e}")
        print(f"Total time:{time.time() - start}")
        if (epoch+1) % args.test_every_n_epochs == 0:
            print("TEST")
            tmp = Procedure.test(dataset, Recmodel, epoch, w, args.multicore, best_result)  # test func
            if tmp['recall'][0] > best_result['recall'][0]:
                best_result = tmp
                torch.save(Recmodel.state_dict(), weight_file)

    print('[phase 2]best result: {recall:', best_result['recall'], 'precision:', best_result['precision'],
          'ndcg:', best_result['ndcg'], 'auc:', best_result['auc'], '}')

finally:
    if args.tensorboard:
        w.close()
