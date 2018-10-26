import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def grad(x: np.ndarray, y: np.ndarray, w=None, k: float = 0.1,
         C: float = 10, e: float = 1e-5, n_max: int = 10000) -> np.ndarray:
    if w is None:
        w = 0
    if isinstance(w, (int, float)):
        w = np.full(len(x[0]), w)
    for n in range(n_max):
        w_old = w
        inner = 1 - 1 / (1 + np.exp(-y * np.sum(x * w_old, axis=1, keepdims=True)))
        w = w_old + k * np.average(y * x * inner, axis=0) - k * C * w_old
        eu = np.linalg.norm(w - w_old)
        # print(f'n={n}\teu={eu}\tw={w}')
        if eu <= e:
            break
    return w


def sigm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.sum(x * w, axis=1, keepdims=True)))


df = pd.read_csv('task3.3.csv', header=None)
X = np.array(df.loc[:, 1:])
Y = np.array(df.loc[:, 0:0])
W = np.zeros(len(X[0]))
g_no = grad(X, Y, W, C=0)
g_l2 = grad(X, Y, W)
s_no = sigm(X, g_no)
s_l2 = sigm(X, g_l2)
print(roc_auc_score(Y, s_no))
print(roc_auc_score(Y, s_l2))
