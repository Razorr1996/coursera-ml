import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV

if __name__ == '__main__':
    newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
    X = newsgroups.data
    Y = newsgroups.target

    vectorizer = TfidfVectorizer()
    X1 = vectorizer.fit_transform(X, Y)

    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, random_state=241, shuffle=True)
    clf = SVC(random_state=241, kernel='linear')
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    gs.fit(X1, Y)

    clf2 = SVC(kernel='linear', random_state=241, C=1.0)
    clf2.fit(X1, Y)
    word_indexes = np.argsort(np.abs(np.asarray(clf2.coef_.todense())).reshape(-1))[-10:]
    words = sorted([vectorizer.get_feature_names()[i] for i in word_indexes])
    print(' '.join(words))
