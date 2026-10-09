import torch
import math

# This module teaches backpropagation.

# The forward pass computes the losses. (This loss is calculated at every step)
# The backwards pass computes the d(loss)/d(weights) via chain rule to compute the actual difference each weight makes to a loss.
# Backpropagation is a way to track the gradients at individual steps and then multiply them all and get back the overall gradient.
# Eg: take an input x, put it through a function a to get y.

# a(x) = y, (repeat this step with a function b)
# b(y) = z.

# Individual gradients are: dy/dx and then dz/dy.
# By the chain rule: dy/dx * dz/dy is the overall gradient.
# Backpropagation tracks all of these gradients and determines WHICH gradient contributed to the MOST loss.

# d(loss)/dx = d(loss)/dz * dz/dx

# Example code:

x = torch.tensor(1.0) # Input Dn
y = torch.tensor(2.0) # Output Dn

weight = torch.tensor(1.0, requires_grad=True) # Defined weight

# Forward pass to compute the loss >>>

y_cap = weight * x # Define a layer that contains that weight and x as a function y_cap
loss = (y_cap - y).square() # The loss function, computed with a square to eliminate errors until they are miniscule.

print(loss)

# Backward pass: computes local gradients and backward pass

loss.backward()
# Calculate the derivative of this loss with respect to every trainable tensor that contributed to producing it.
# And store those derivatives in their .grad attributes.

print(weight.grad)

# Next step would be updating the weights, and redoing this step for a number of iterations until loss is minimized.





