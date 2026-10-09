import torch
import numpy as np

x = torch.rand(4, 4)
print(x)
y = x.numpy() # You can use this to represent a pytorch tensor as a numpy array.
print(y) # By default, the data type for the array if float64 or 64 floating point digits.

# Note that both the pytorch tensor and the numpy array both point to the same location in memory.
# Therefore, if you change something in the tensor even after declaring the numpy array.
# the numpy array will still reflect that same change.

# Conversely...

a = np.ones((4, 4))
print(a)
b = torch.from_numpy(a) # This changes the numpy array to a pytorch tensor.
print(b)

# It is possible to manipulate tensors directly on and from the gpu if you have a dedicated GPU and CUDA available.

if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.rand(3, 3, device=device) # This creates memory space directly on the GPU...
    # And manipulates the tensors directly from there, resulting in lightning fast calculations for large tensors.
    y = x.to(device) # this piece of code transfers the gpu tensor to the cpu.

l = torch.ones(5, requires_grad=True) # This line of code is used for optimizing the tensors.
# It is usually used for when you will have variables in the tensor. It declares the gradient for the tensor.



