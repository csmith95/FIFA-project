import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

clean = pd.read_csv('data/2019clean.csv', index_col=0)
value = pd.read_csv('data/2019.csv', index_col=0)

merged = clean.merge(value.loc[:, ['ID', 'Value']], on='ID')

print(merged.head())


merged.Value = [(float(v[1:-1]) if v[-1] != '0' else 0.0) * (1000000 if v[-1] == 'M' else 1000) for v in merged.Value]

print(merged.columns)
X = merged.drop(['ID', 'Value', 'Name'], axis=1)
y = merged.Value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

dev_predictions = linear_regression.predict(X_dev)
print("Baseline model mean squared error: %.2e" % mean_squared_error(y_dev, dev_predictions))
print("Naive mean squared error: %.2e" % mean_squared_error(y_dev, [np.mean(y_train)] * len(y_dev)))


