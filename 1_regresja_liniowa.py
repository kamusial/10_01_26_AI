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

