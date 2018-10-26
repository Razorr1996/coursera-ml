import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def female_name(s: str) -> str:
    if 'Mrs.' in s:
        pass
    else:
        pass
    return ''


df = pd.read_csv('task1.1.csv', index_col='PassengerId')
df2 = df[df.Sex == 'female'][['Name']]
d = set()

dt = {'Deg': [], 'Name':[]}
for i in df2.Name:
    a = list(i.split(sep=','))[1].split('.')
    dt['Deg'].append(a[0][1:])
    dt['Name'].append('. '.join(a[1:]))
df3 = pd.DataFrame(dt)

names = []
for i, e in df3.iterrows():
    name = ''
    if any(i in e.Deg for i in ['Miss', 'Mlle', 'Mme', 'Ms', 'Dr']):
        name = e.Name.split()[0]
    elif any(i in e.Deg for i in ['Mrs', 'Lady', 'the Countess']):
        if '(' in e.Name:
            name = e.Name.split('(')[1].split()[0]
        else:
            name = e.Name.split()[0]
    else:
        print(e)
    for j in ['(', ')']:
        name = name.replace(j, '')
    names.append(name)

npd = pd.Series(sorted(names))
print(npd.value_counts())
