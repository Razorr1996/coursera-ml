import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('task3.1.csv', header=None)
X = df.loc[:, 1:]
Y = df.loc[:, 0:0]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X,Y)
print(clf.support_)
