import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

clean = pd.read_csv('data/2019clean.csv', index_col=0)
value = pd.read_csv('data/2019.csv', index_col=0)

merged = clean.merge(value.loc[:, ['ID', 'Value']], on='ID')

print(merged.head())


merged.Value = [(float(v[1:-1]) if v[-1] != '0' else 0.0) * (1000000 if v[-1] == 'M' else 1000) for v in merged.Value]

X = merged.drop(['ID', 'Value', 'Name'], axis=1)
y = merged.Value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# Linear regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
dev_predictions_linear = linear_regression.predict(X_dev)
print("Linear regression model mean squared error: %.2e" % mean_squared_error(y_dev, dev_predictions_linear))
print("Naive mean squared error: %.2e" % mean_squared_error(y_dev, [np.mean(y_train)] * len(y_dev)))

# Ridge regression
ridge_regression = Ridge(alpha=0.5)
ridge_regression.fit(X_train, y_train)
dev_predictions_ridge = ridge_regression.predict(X_dev)
print("\nRidge regression model mean squared error: %.2e" % mean_squared_error(y_dev, dev_predictions_ridge))
print("Naive mean squared error: %.2e" % mean_squared_error(y_dev, [np.mean(y_train)] * len(y_dev)))

# Support vector regression
params = {'C': sp.stats.expon(scale=100), 'gamma': sp.stats.expon(scale=.1), 'kernel': ['rbf']}
svr = SVR()
random_search = RandomizedSearchCV(svr, param_distributions=params, n_iter=100, cv=5, iid=False)
random_search.fit(X_train, y_train)
dev_predictions_svr = random_search.best_estimator_.predict(X_dev)
print("\nSVR model mean squared error: %.2e" % mean_squared_error(y_dev, dev_predictions_svr))
print("Naive mean squared error: %.2e" % mean_squared_error(y_dev, [np.mean(y_train)] * len(y_dev)))


