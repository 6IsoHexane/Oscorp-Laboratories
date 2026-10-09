import torch
import numpy as np

# The gradient of a tensor is the partial derivative of each of its individual components.
# i.e for a function say f(x) = x^2 + y^2, its tensor say: [x^2, y^2], the gradient would be d(f(x))/dx and d(f(x))/dy.
# Therefore gradient = [2x, 2y]
# A gradient describes how that scalar changes with each tracked input or parameter. (How each changing parameter affects the result as a whole)

x = torch.rand(3, requires_grad=True)
print(x)
# The requires gradient function tells PyTorch to remember the original function.
# PyTorch will then allocate some memory to remember all the operations it goes through, maintaining a log of whatever has happened.
# After which, pytorch will automatically calculate the partial derivatives of all the elements within the function, represented as a tensor.

y = x+2
print(y)
# The result will show the gradient of the tensor, i.e backtrack and tell you what exactly happened to it before spitting out an answer.
# For this one, the grad_fn will be "AddBackward0"

z = y*y*2
print(z)
# Z will have the gradient "MulBackward0"
z = z.mean()
print(z) # (13.7291, grad_fn=<MeanBackward0>), Gives this.

z.backward()
# This will give you dz/dx
# Note that this function only and only works for scalar values, not vectors. If you have to make a scalar value of it, always multiply it by some other vector.

print(x.grad)

# When you want the program to stop tracking the gradient...
# x.requires_grad_(False)



# Some example code with uses of this:

some_weight = torch.ones(3, requires_grad=True)
print(some_weight)

for epoch in range (1): # For _ iterations...

    model_output = (some_weight*3).sum() # New scalar, model_output = tensor (some_weights*3)... Gives you a SCALAR value.

    model_output.backward() # Tracks back to see which operations were performed... Without this, the gradient won't be ascertained.

    print(some_weight.grad) # And gives you the partial derivatives, taking into consideration the number of iterations.

# You can later empty/delete the variables by running the following piece of code:

some_weight.grad.zero_() # This sets the gradient to zero.
print(some_weight.grad)


# Yet another example:
# If you make an optimizer for some tensor...

weight_a = torch.ones(3, requires_grad=True)

optimizer = torch.optim.SGD([weight_a], lr=0.01)
# What this does is...
# It finds the loss between the actual and the predicted using the loss function.
# It finds out how much has the value changed using the data from the gradient.
# Depending on the gradient and the optimizer step you choose, it makes changes to known training weights.

optimizer.step()
optimizer.zero_grad()
# This last line flushes the gradient from the last iteration so that we arent using the same old gradients during the optimization steps.


## Final take: Always use the requires_grad function/parameter whenever you need to calculate gradients.
## So... Always.