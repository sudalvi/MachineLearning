import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")
print(data.shape)
data.head()

X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncode = LabelEncoder()
X[:,3] = labelEncode.fit_transform(X[:,3])
oneHotEncode = OneHotEncoder(categorical_features=[3])
X = oneHotEncode.fit_transform(X).toarray

X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=1/2, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)

y_pred = reg.predict(X_test)
print(y_pred)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()

X_opt = X[:,[0,3,4,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()

X_opt = X[:,[0,3,5]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()

X_opt = X[:,[0,3]]
reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
reg_OLS.summary()



'''states = pd.get_dummies(X['State'],drop_first=True)

X = X.drop('State', axis=1)

X = pd.concat([X,states], axis = 1)
X = X.astype(int)
X = X.astype(int)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
X_train = np.arange(min(X), max(X), 0.01)
Y_train = np.arange(min(Y), max(Y), 0.01)
X_train = X_train.reshape((len(X_train), 1))
Y_train = Y_train.reshape((len(Y_train), 1))

plt.scatter(X_train, Y_train, color='r', label="SS")
plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train,Y_train)

y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(Y_test, y_pred)
print(score)'''
