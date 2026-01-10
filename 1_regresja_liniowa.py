import time
import pandas as pd

# time.sleep(2)
print('Program')

df = pd.read_csv('weight-height.csv', sep=';')       # czytanie pliku csv
print(df)
# print(type(df))

# df.Height = df.Height * 2.54
df.Height *= 2.54
# df.Weight = df.Weight / 2.2
df.Weight /= 2.2

print('\nPo zmianie jednostek')
print(df)
print(f'ksztalt danych {df.shape}')
print('\nDescribe')
print(df.describe().T.round(2).to_string())

print(f'\nZliczanie wartości kolumny Gender:\n{df.Gender.value_counts()}')

import matplotlib.pyplot as plt
import seaborn as sns

# plt.hist(df.query("Gender=='Male'").Weight)
# plt.hist(df.query("Gender=='Female'").Weight)
# plt.show()
#
# sns.histplot(df.query("Gender=='Male'").Weight)
# sns.histplot(df.query("Gender=='Female'").Weight)
# plt.show()

# zamiana gender na dane numeryczne
df = pd.get_dummies(df)
print(df)

# usunięcie kolumny "Gender_Male"
del (df['Gender_Male'])

# zmiana nazwy kolumny
df = df.rename(columns={'Gender_Female': 'Gender'})
# False - Facet, True - Kobieta
print(df)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit( df[['Height','Gender']], df['Weight'] )
print(f'Współczynnik kierunkowy: {model.coef_}\nWyraz wolny: {model.intercept_}')
print(f'Weight = Height * {model.coef_[0]} + Gender * {model.coef_[1]} + {model.intercept_}')

