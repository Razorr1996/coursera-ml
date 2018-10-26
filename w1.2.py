import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('task1.2.csv', index_col='PassengerId')
df2 = df[df.Age.notnull()][['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
df2.Sex = pd.Series([1 if i == 'male' else 0 for i in df2.Sex], index=df2.index)

clt = DecisionTreeClassifier(random_state=241)
clt.fit(df2[['Pclass', 'Fare', 'Age', 'Sex']], df2.Survived)
print(clt.feature_importances_)
