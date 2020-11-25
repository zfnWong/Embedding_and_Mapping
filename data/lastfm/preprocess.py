import pandas as pd
import numpy as np

trainData = pd.read_table('./train.txt', header=None).to_numpy(dtype=np.int)[:, :2] - 1
testData = pd.read_table('./test.txt', header=None).to_numpy(dtype=np.int)[:, :2] - 1

trainData = trainData[np.argsort(trainData[:, 0])].tolist()
testData = testData[np.argsort(testData[:, 0])].tolist()

train_list = []
current_user = -1
for interaction in trainData:
    if current_user != interaction[0]:
        train_list.append(interaction)
        current_user = interaction[0]
    else:
        train_list[-1].append(interaction[1])

test_list = []
current_user = -1
for interaction in testData:
    if current_user != interaction[0]:
        test_list.append(interaction)
        current_user = interaction[0]
    else:
        test_list[-1].append(interaction[1])

with open('./train1.txt', 'w') as f:
    for row in train_list:
        for item in row:
            f.write(str(item) + '\t')
        f.write('\n')

with open('./test1.txt', 'w') as f:
    for row in test_list:
        for item in row:
            f.write(str(item) + '\t')
        f.write('\n')


