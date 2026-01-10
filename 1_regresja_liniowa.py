import time
import pandas as pd

# time.sleep(2)
print('Program')

df = pd.read_csv('weight-height.csv',sep=';')       # czytanie pliku csv
print(df)
