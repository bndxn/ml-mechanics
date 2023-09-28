import numpy as np
import math 
from sklearn.linear_model import LogisticRegression

x = np.array([0.8, 0.2, 0.6, 0.95, -0.4, -0.5])
y = np.array([1, 0, 1, 1, 0, 0])

def update_parameters(gradient, learning_rate, start, n_iter, tolerance):
    
    """
    This function updates the vector following the updating rule. 

    Args:
        - gradient: a function which calculates the gradient of our loss function
        - learning_rate: the amount that the parameter values change each step
        - start: the randomly initialised values for the slope and intercept
        - n_iter: the number of steps to run through 
    """

    # Our starting value
    vector = start

    # we do n_iter passes through, each time working out the gradient of the loss function
    # because we know the loss function, we can work out the gradient

    for _ in range(n_iter):
        update = -learning_rate * np.array(gradient(x, y, vector))
        vector += update
    print(f'Slope {vector[1]}, intercept: {vector[0]}')
    return vector


def sigmoid_predictions(vector, x):
    
    predictions = 1 / (1 + math.e ** -(vector[0] + vector[1]*x))
    return predictions

def binary_cross_entropy_loss_gradient(x, y, vector):

    # While we do not know the values of the parameters, we do know our loss function
    # Our loss function is binary cross entropy
    # Taking the derivative of the loss function w.r.t. x just leaves y - predicitions
    # So we broadcast this across a numpy array, x of inputs, y is predictions

    # trying some L2 norming
    error = -(y - sigmoid_predictions(vector, x))
    print(f'Calculating gradient. Total residuals: {np.round(error.sum(),2)}') 

    return error.mean(), (error * x).mean()

update_parameters(binary_cross_entropy_loss_gradient, learning_rate=0.001, 
                  start=[0, 0], n_iter=10000, tolerance=1e-2)


x_array = x.reshape(-1,1)
clf = LogisticRegression(penalty='l2').fit(x_array,y)
#print(x_array, y)
print(f'Sklearn - slope:{clf.coef_}, intercept: {clf.intercept_}')
