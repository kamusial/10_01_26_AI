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
    df[col] = df[col].replace(0, np.nan)
    mean_ = df[col].mean()
    df[col].replace(np.nan, mean_, inplace=True)  # przestanie działać

print('\nPo czyszczeniu danych:')
print(df.describe().T.to_string())
print(df.isna().sum())    # suma pustych pól
df.to_csv('data\\cukrzyca_wyczyszczone_dane.csv', sep=';', index=False)

X = df.iloc[:, :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('sprawdżmy, czy klasy zbalansowane')
print(df.outcome.value_counts())

print('zmiana danych')
df1 = df.query('outcome==0').sample(n=500)
df2 = df.query('outcome==1').sample(n=500)
df3 = pd.concat([df1, df2])   #łączenie

X = df3.iloc[:, :-1]
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

# KNN dla porównania
print('\nKNN')
X = df.iloc[:, :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=50, weights='distance')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))