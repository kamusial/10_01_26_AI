import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data\\diabetes.csv')
print(f'Ksztalt danych: {df.shape}')
print(df.describe().T.to_string())
print(f'\nPuste wartosci:')
print(df.isna().sum())    # ile pustych wartości w kolumnach

# wszędzie, gdzie zera i puste wartości - zastąp wrtością średnią średnią (bez zer)
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.nan, inplace=True)
    mean_ = df[col].mean()
    df[col].replace(np.nan, mean_, inplace=True)
