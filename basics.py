import torch

x = torch.empty(1)
print(x)
# The above code has declared x as an empty tensor. The tensor is one dimensional.

y = torch.empty(3)
print(y)
# This piece of code has declared y as another tensor. This time, the tensor is STILL 1 dimensional, but it has three elements.

z = torch.rand(3, 3)
print(z)
# This piece of code has now created a 3*3, random tensor, or a 3*3 matrix.

alpha = torch.rand(3, 3, 3)
print(alpha)
# This piece of code has declared alpha, a 3*3*3 tensor, or a tensor in three dimensions, 3 units each way.
# Although this vector exists, it is hard to visualize on screen.

a =  torch.zeros(2, 2)
# This will create a 2*2 matrix of just zeros.

b = torch.ones(2, 2)
# This will create a 2*2 matrix of all ones.

# D_TYPE PARAMETER

h = torch.ones(3, 3, dtype=torch.int) # D-type is a parameter that stands for "data type"
print(h.dtype) # The data type can be specified like how it is done in general. The parameter has to be mentioned when using it in any statement.
print(h.size()) # YOu can use the size function to check the size of a tensor.

i = torch.tensor([1, 2, 3]) # the torch.tensor statement is used to declare custom tensors.
# it can be used in the form of lists where your elements are within box brackets.