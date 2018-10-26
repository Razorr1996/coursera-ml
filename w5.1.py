import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv('task5.1.csv')
df.Sex = df.Sex.map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = df.loc[:, 'Sex': 'ShellWeight']
Y = df.loc[:, 'Rings']

neighbors = 50
kf = KFold(n_splits=5, shuffle=True, random_state=1)

mean = dict()
for n in range(1, neighbors + 1):
    rfr = RandomForestRegressor(random_state=1, n_estimators=n)
    cv = cross_validate(rfr, X, Y, cv=kf, scoring=make_scorer(r2_score), n_jobs=4)
    t = cv['test_score'].mean()
    mean[n] = t
    print(f'n={n};\tscore={t}')
    if t > 0.52:
        break
