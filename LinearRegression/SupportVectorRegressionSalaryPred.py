import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:,1:2].values
Y = data.iloc[:, 2].values
Y = Y.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
st_y = StandardScaler()
X = st_x.fit_transform(X)
Y = st_y.fit_transform(Y)

from sklearn.svm import SVR
regression = SVR(kernel='rbf')
regression.fit(X, Y)

y_pred = regression.predict(X)

plt.scatter(X,Y, color='b')
plt.plot(X, y_pred, color='r')
plt.title("Support Vector Regression For Salary Prediction")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
