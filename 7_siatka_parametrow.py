import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('heart.csv', comment='#')
print(df.head().to_string())
print(df.target.value_counts())

X = df.iloc[:, :-1]
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
# max_depth int, default=None
# min_samples_split int or float, default=2
# max_features int, float or {“sqrt”, “log2”}, default=None

model = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=2)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))

print('\nSiatka parametrów')
model = DecisionTreeClassifier()
params = {
    'max_depth': range(3, 14),  # od 3 do 13 z krokiem 2
    'max_features': range(3, X_train.shape[1]+1),
    'min_samples_split': [2, 3, 4, 5],
    'criterion': ['gini', 'entropy', 'log_loss']
}
grid = GridSearchCV(model, params, scoring="accuracy", cv=10, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

