import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import scale


boston = load_boston()
X = scale(boston.data)
Y = boston.target
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mean_p = dict()
for n in np.linspace(1,10,200):
    knp = KNeighborsRegressor(n_neighbors=5, weights='distance', p=n)
    cv = cross_validate(knp, X, Y, cv=kf, scoring='neg_mean_squared_error')
    mean_p[n] = cv['test_score'].mean()
m = sorted(mean_p.items(), key=lambda x: x[1], reverse=True)
print(m[0])
