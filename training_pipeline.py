import torch
import numpy as np

# In this module is developed, simple ML algorithm explaining all the basics.

### STEP 1, a simple algo using numpy.

x = np.array([1, 2, 3, 4], dtype=np.float32) # Input training data x(i)
y = np.array([2, 4, 6, 8], dtype=np.float32) # Output training data y(i)

# Therefore, there is some weight w such that the function y = w * x must be true.
# By mere observation, it is evident that this weight is the scalar 2.
# But the machine doesn't know that... The machine is going to learn that weight.
# This, is the core principle of machine learning.

w = 0.0 # We initiliaze our weight to 0 to start off with.

# Now we define the forward pass:
def forward(x):
    return w * x

# Now we define the loss function
def loss_fn(y_predicted, y):
    return (y_predicted - y).square().mean()

# Gradient of the loss:
# The mean square error, MSE = 1/n (y_predicted - y).square() (Essentially, the mean of the square of the errors)
# The gradient itself = 1/n (y_predicted - y).square() * 2x (Counts the derivative in to calculate th gradient descent.)
def gradient_descent(x, y, y_predicted):
    return np.dot(2x, y_predicted-y).mean()
    # The gradient descent is the mean of the dot product between the derivative and y_predicted-y.






