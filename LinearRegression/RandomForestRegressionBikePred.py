import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bike_sharing.csv')
X = data.iloc[:,-7].values
Y = data.iloc[:,-1].values

X = X.reshape(-1,1)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_train)

X_grid = np.arange(min(X_train), max(X_train), 0.001)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_train,Y_train, color='r')
plt.plot(X_grid,regressor.predict(X_grid), color='b')
plt.title("Decision Tree For Bike Prediction")
plt.xlabel("Temp")
plt.ylabel("Count")
plt.legend()
plt.show()