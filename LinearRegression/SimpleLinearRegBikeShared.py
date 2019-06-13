import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("bike_sharing.csv")
print(data.shape)
data.head()

X = data.iloc[:,-7].values
Y = data.iloc[:,-1].values

X = X.reshape(-1,1)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=9, random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression = regression.fit(X_train, Y_train)
print(regression.score(X_train, Y_train))
y_pred = regression.predict(X_test)
print('y_pred' , y_pred)

plt.scatter(X_train, Y_train, color='b', label='Scatter Plot')
plt.plot(X_train, regression.predict(X_train), color='r', label='Regression Line')
plt.title('Bikes Shared based on temperature')
plt.xlabel('Temp')
plt.ylabel('Count')
plt.legend()
plt.show()

