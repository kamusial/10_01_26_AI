import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data\\otodom.csv')
print(df)
print(type(df))

print(df.describe().T.to_string())

print(df.iloc[:, 1:].corr())  # korelacja między parametrami, bez pierwszej kolumny
sns.heatmap(df.iloc[:, 1:].corr(), annot=True)
plt.show()

sns.histplot(df.price)
plt.show()

q1 = df.describe().T.loc['price', '25%']
q3 = df.describe().T.loc['price', '75%']

df1 = df[(df.price >= 0) & (df.price <= q3)]

sns.histplot(df1.price)
plt.show()

X = df1.iloc[:, 2:]
y = df1.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

print(pd.DataFrame(model.coef_, X.columns))