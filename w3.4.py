import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve

dfc = pd.read_csv('task3.4_class.csv')
dfs = pd.read_csv('task3.4_scores.csv')

ndfc = np.array(dfc)
c = dict(zip(*np.unique(np.sum(ndfc * np.array([2, 1]), axis=1), return_counts=True)))
TP = c[3]
FN = c[2]
FP = c[1]
TN = c[0]
print(' '.join(map(str, [TP, FP, FN, TN])))

acc = accuracy_score(dfc.true, dfc.pred)
prec = precision_score(dfc.true, dfc.pred)
rec = recall_score(dfc.true, dfc.pred)
f = f1_score(dfc.true, dfc.pred)
print(' '.join(map(str, [acc, prec, rec, f])))

roc = np.zeros(4)
ndfs = np.array(dfs)
for i in range(4):
    roc[i] = roc_auc_score(ndfs[:, 0], ndfs[:, i + 1])

for i in range(4):
    t4 = np.array(list(zip(*precision_recall_curve(ndfs[:, 0], ndfs[:, i + 1]))))
    t = t4[t4[:, 1] >= 0.7, :]
    m = np.amax(t[:, 0])
    print(f'{i} {m}')
