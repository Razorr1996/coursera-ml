import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, coo_matrix

train = pd.read_csv('task4.2_prices.csv')
dj = pd.read_csv('task4.2_djia.csv')

X = train.loc[:, 'AXP':]

p = PCA(n_components=10)

p.fit(X)

c1 = p.transform(X)[:, 0]
