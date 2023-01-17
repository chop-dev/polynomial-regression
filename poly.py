import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[1], [3], [3], [4], [5], [6], [6], [8], [9], [10]] # age of bluegill in years
y_train = [[1], [5], [8], [11], [23], [38], [46], [62], [83], [100]] # length of bluegill in mm

# Testing set
x_test = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]] # age of bluegill in years
y_test = [[1], [4], [9], [12], [25], [36], [49], [64], [81], [100]] # length of bluegill in mm

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 10, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree = 2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c = 'r', linestyle = '--')
plt.title('Length vs Age')
plt.xlabel('Age of bluegill in years')
plt.ylabel('Length of bluegill in mm')
plt.axis([0, 10, 0, 100])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print(x_train)
print(x_train_quadratic)
print(x_test)
print(x_test_quadratic)
