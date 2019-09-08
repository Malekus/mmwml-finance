import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


data = pd.read_csv("NFLX.csv", delimiter=",")
dataNumpy = data.to_numpy()

X = dataNumpy[:, 0]
y = dataNumpy[:, 5]


plt.figure()
plt.plot(X, y)
plt.xticks([date for index, date in enumerate(X) if index % 25 == 0 or index == len(X)-1])
plt.grid(axis='x')
plt.show()


# Moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#Find best moving average
r = []
for n in range(2, 20):
    r.append(mean_squared_error(dataNumpy[n-1:, 5], moving_average(dataNumpy[:, 5], n)))
    
plt.figure()
plt.plot([i for i in range(2, 20)], r)
plt.scatter([i for i in range(2, 20)], r, marker='x')
plt.xticks([i for i in range(2, 20)])
plt.show()

predict = moving_average(dataNumpy[:, 5], 2)

plt.figure()
plt.title('Moving Average')
plt.plot(X, y, c='red')
plt.scatter(X, y, c='red', marker='x')
plt.plot(X[1:], predict, c='blue')
plt.scatter(X[1:], predict, c='blue', marker='x')
plt.xticks([date for index, date in enumerate(X) if index % 25 == 0 or index == len(X)-1])
plt.grid(axis='x')
plt.legend(['test', 'predict'])
plt.show()



X_train, X_test, y_train, y_test = train_test_split(dataNumpy[:, 1], y, test_size=0.4, random_state=0)

clf = LinearRegression()  
clf.fit(X_train.reshape(-1, 1), y_train)
y_pred = clf.predict(X_test.reshape(-1, 1))

plt.figure()
plt.title('LinearRegression')
plt.scatter(X_test, y_test,  color='red', marker='x')
plt.plot(X_test, y_pred, color='blue', linewidth=1)
plt.legend(['predict', 'test'])
plt.show()


clf2 = Ridge()
clf2.fit(X_train.reshape(-1, 1), y_train)
y_pred2 = clf2.predict(X_test.reshape(-1, 1))

plt.figure()
plt.title('Ridge')
plt.scatter(X_test, y_test,  color='red', marker='x')
plt.plot(X_test, y_pred2, color='blue', linewidth=1)
plt.legend(['predict', 'test'])
plt.show()


clf3 = Lasso(alpha=0.1)
clf3.fit(X_train.reshape(-1, 1), y_train)
y_pred3 = clf3.predict(X_test.reshape(-1, 1))

plt.figure()
plt.title('Lasso')
plt.scatter(X_test, y_test,  color='red', marker='x')
plt.plot(X_test, y_pred3, color='blue', linewidth=1)
plt.legend(['predict', 'test'])
plt.show()







