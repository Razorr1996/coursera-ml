import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('task2.3_train.csv', header=None)
df_test = pd.read_csv('task2.3_test.csv', header=None)

scaler = StandardScaler()

X_train = df_train.loc[:, 1:]
Y_train = df_train.loc[:, 0:0]

X_test = df_test.loc[:, 1:]
Y_test = df_test.loc[:, 0:0]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

ac = accuracy_score(Y_test, predictions)

clf2 = Perceptron(random_state=241)
clf2.fit(X_train_scaled, Y_train)
predictions2 = clf2.predict(X_test_scaled)

ac2 = accuracy_score(Y_test, predictions2)

print(f'ac:{ac}\tac2:{ac2}\tdelta:{ac2-ac}')
