import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, coo_matrix

train = pd.read_csv('task4.1_train.csv')
test = pd.read_csv('task4.1_test.csv')

enc = DictVectorizer()
vec = TfidfVectorizer(min_df=5)

train.FullDescription = train.FullDescription.str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
test.FullDescription = test.FullDescription.str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

X1 = vec.fit_transform(train.FullDescription)
X1_test = vec.transform(test.FullDescription)

train.LocationNormalized.fillna('nan', inplace=True)
train.ContractTime.fillna('nan', inplace=True)

X2 = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X2_test = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

r = Ridge(alpha=1, random_state=241)

xxx = hstack([X1, X2])
xxx_test = hstack([X1_test, X2_test])

r.fit(xxx, train.SalaryNormalized)
print(r.predict(xxx_test))
