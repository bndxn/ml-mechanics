"""
linreg_backpropagation.py

Useful links:
- https://realpython.com/gradient-descent-algorithm-python
"""
import numpy as np

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])

# The updating rule looks like this:
# v -> v - \eta \grad v
# v: vector
# eta: learning rate
# grad v -> the gradient vector, including all the partial derivatives

def update_parameters(gradient, learning_rate, start, n_iter):
    
    """
    This function updates the vector following the updating rule. 

    We need to pass in 
    """

    # Our starting value
    vector = start

    # we do n_iter passes through, each time working out the gradient of the loss function
    # because we know the loss function, we can work out the gradient

    for _ in range(n_iter):
        update = -learning_rate * np.array(gradient(x, y, vector))
        vector += update
        print(vector)

    print(vector)
    return vector


def sum_of_squares_gradient(x, y, vector):

    # While we do not know the values of the parameters, we do know our loss function
    # Our loss function is sum of squares, and we want to work out the loss function w.r.t both parameters

    # Cost = sum(y_i - b_0 - b_1 x_i)^2 / n
    # We want the partial derivatives of this w.r.t b_0 and b_1
    # dCost/db_0 = 2 sum(y_i - b_0 - b_1 * x_i) / n
    # dCost/db_1 = 2 sum(y_i - b_0 - b_1 * x_i) * x_i / n

    # We ignore the factor of on both since this just scales the vector and doesn't change the direction

    residuals = vector[0] + vector[1] * x - y
    print(f'Calculating gradient. Total residuals: {residuals.sum()}') 
    return residuals.mean(), (residuals * x).mean()

update_parameters(sum_of_squares_gradient, learning_rate=0.01, start=[0.5, 0.5], n_iter=100)


