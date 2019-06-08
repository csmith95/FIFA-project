import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

clean_2018 = pd.read_csv('data/2018clean.csv', index_col=0)
clean_2019 = pd.read_csv('data/2019clean.csv', index_col=0)
value = pd.read_csv('data/2019.csv', index_col=0)

merged = clean_2019.merge(value.loc[:, ['ID', 'Value']], on='ID')

print(merged.head())

merged.Value = [(float(v[1:-1]) if v[-1] != '0' else 0.0) * (1000000 if v[-1] == 'M' else 1000) for v in merged.Value]

name_to_rating_2018 = dict(zip(clean_2018['ID'], clean_2018['Overall']))
name_to_rating_2019 = dict(zip(clean_2019['ID'], clean_2019['Overall']))


################## Predicting Player Value Experiments ###################

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

# NN
hSizes = [(100), (100, 150, 50), (150, 100, 50)] 
for h in hSizes:
	nn_model = MLPRegressor()
	nn_model.fit(X_train, y_train)
	dev_predictions_nn = nn_model.predict(X_dev)
	print("\nNN model mean squared error: %.2e" % mean_squared_error(y_dev, dev_predictions_nn))
	print("Hidden layer sizes: {}".format(h))
	print("Naive mean squared error: %.2e" % mean_squared_error(y_dev, [np.mean(y_train)] * len(y_dev)))

################## Predicting Player Improvement Trajectory Experiments ###############################
print("\n\n" + "*"*10 + "Player Improvement Experiments" + "*"*10)

for player in clean_2018['ID']:
	if name_to_rating_2018[player] < name_to_rating_2019[player]:
		y_improvement.append(1)
	else:
		y_improvement.append(0)

X_imp_train, X_imp_test, y_imp_train, y_imp_test = train_test_split(X, y_improvement, test_size=0.2, random_state=1)
X_imp_dev, X_imp_test, y_imp_dev, y_imp_test = train_test_split(X_imp_test, y_imp_test, test_size=0.5, random_state=1)

# Support vector regression
# params = {'C': sp.stats.expon(scale=100), 'gamma': sp.stats.expon(scale=.1), 'kernel': ['rbf']}
# svr = SVR()
# random_search = RandomizedSearchCV(svr, param_distributions=params, n_iter=100, cv=5, iid=False)
# random_search.fit(X_train, y_train)
# dev_predictions_svr = random_search.best_estimator_.predict(X_dev)
# print("\nSVR model mean squared error: %.2e" % mean_squared_error(y_dev, dev_predictions_svr))
# print("Naive mean squared error: %.2e" % mean_squared_error(y_dev, [np.mean(y_train)] * len(y_dev)))

# Logistic regression
logistic_regression = LogisticRegression(solver='lbfgs', max_iter=500)
logistic_regression.fit(X_imp_train, y_imp_train)
dev_predictions_logistic = logistic_regression.predict(X_imp_dev)
print("Logistic regression model mean squared error: %.2e" % mean_squared_error(y_imp_dev, dev_predictions_logistic))
print("Naive mean squared error: %.2e" % mean_squared_error(y_imp_dev, [np.mean(y_imp_train)] * len(y_imp_dev)))

# NN 
hSizes = [(100), (100, 150, 50), (150, 100, 50)] 
for h in hSizes:
	nn_model = MLPClassifier()
	nn_model.fit(X_imp_train, y_imp_train)
	dev_predictions_nn = nn_model.predict(X_dev)
	print("\nNN model mean squared error: %.2e" % mean_squared_error(y_imp_dev, dev_predictions_nn))
	print("Hidden layer sizes: {}".format(h))
	print("Naive mean squared error: %.2e" % mean_squared_error(y_imp_dev, [np.mean(y_train)] * len(y_dev)))

