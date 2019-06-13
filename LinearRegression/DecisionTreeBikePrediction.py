import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bike_sharing.csv')
X = data.iloc[:,-7].values
Y = data.iloc[:,-1].values
print(X)
print(Y)
X = X.reshape(-1,1)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,Y_train)

X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X_train, Y_train, color='b')
plt.plot(X_grid, regressor.predict(X_grid), color='r')
plt.title("Decision Tree For Bike Prediction")
plt.xlabel("Temp")
plt.ylabel("Count")
plt.show()