import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")
print(data.shape)
data.head()

X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)
y_pred = regression.predict(X_test)
plt.scatter(X_test, Y_test, color='b', label='Scatter Plot')
plt.plot(X_train, regression.predict(X_train), color='r', label='Regression Plot')
plt.title('Prediction Salary Based On Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

