import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import scale

neighbors = 50
df = pd.read_csv('task2.1.csv')
X = df.loc[:, 'Alcohol':]
Y = df.loc[:, 'Class']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mean_k1 = dict()
for n in range(1, neighbors + 1):
    knc = KNeighborsClassifier(n_neighbors=n)
    cv = cross_validate(knc, X, Y, cv=kf, scoring='accuracy')
    mean_k1[n] = cv['test_score'].mean()
m1 = sorted(mean_k1.items(), key=lambda x: x[1], reverse=True)
print(m1[0])

X1 = scale(X)
mean_k2 = dict()
for n in range(1, neighbors + 1):
    knc = KNeighborsClassifier(n_neighbors=n)
    cv = cross_validate(knc, X1, Y, cv=kf, scoring='accuracy')
    mean_k2[n] = cv['test_score'].mean()
m2 = sorted(mean_k2.items(), key=lambda x: x[1], reverse=True)
print(m2[0])
