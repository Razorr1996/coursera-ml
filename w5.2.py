import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def f_loss(y_true, y_pred):
    y2 = 1 / (1 + np.exp(-y_pred))
    return log_loss(y_true, y2)


df = pd.read_csv('task5.2.csv')
ndf = df.values

X = ndf[:, 1:]
Y = ndf[:, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=241)

lrs = [1, 0.5, 0.3, 0.2, 0.1]
# lrs = [0.2]

for lr in lrs:
    plt.figure()
    clf = GradientBoostingClassifier(random_state=241, n_estimators=250, learning_rate=lr, verbose=True)
    clf.fit(X_train, Y_train)
    loss_train = np.array([f_loss(Y_train, i) for i in clf.staged_decision_function(X_train)])
    loss_test = np.array([f_loss(Y_test, i) for i in clf.staged_decision_function(X_test)])
    mn = np.argmin(loss_test)
    mz = loss_test[mn]
    print(f'{np.around(mz,2)} {mn+1}')

    plt.plot(loss_train, 'g', linewidth=2)
    plt.plot(loss_test, 'r', linewidth=2)
    plt.legend(['train', 'test'])
    plt.title(f'learning_rate {lr}')
    plt.show()

clf2 = RandomForestClassifier(n_estimators=37, random_state=241)
clf2.fit(X_train, Y_train)
pp = clf2.predict_proba(X_test)
print(log_loss(Y_test, pp[:,1]))
