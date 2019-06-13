import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:2].values
Y = data.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,Y)

y_pred = regressor.predict(6.5)

X_grid = np.arange(min(X), max(X), 0.01)
print('first',X_grid)
print('len(X_grid)',len(X_grid))
X_grid = X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,regressor.predict(X_grid), color='b')
plt.title("Decision Tree For Salary Prediction")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()