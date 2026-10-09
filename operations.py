import torch

simple_addition = False
in_Place_addition = False
multiplication = False
slicing = False
resizing = True

if simple_addition:
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    print(x)
    print(y)

    # Two tensors can be added the following ways:

    z = x + y
    print(z)
    # Or alternatively...
    z = torch.add(x, y)  # PyTorch has a built-in function for adding two tensors together.

if in_Place_addition:
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    y.add_(x) # This adds the value of x to y. i.e, the value of y is now changed.
    print(y)

if multiplication:
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    z = torch.mul(x, y) # This function multiplies the two tensors together.
    w = torch.div(x, y) # This one is used to divide tensor x by tensor y.
    print(z)

if slicing:
    x = torch.rand(2, 2)
    print(x[:, 0]) # This prints all the columns of the first row, index 0)
    print(x[1, 1]) # Prints a(1, 1)

if resizing:
    x = torch.rand(4, 4)
    print(x)
    y = x.view(16) # This, reshapes our tensor from a 4*4 view to a 1*16 view.
    print(y)
    z = x.view(16, 1) # This just turned it into a 16*1 view
    print(z)
    a = x.view(-1, 9) # If you don't know or can't specify on dimension when reshaping a vector...
    # just put a -1 and a positive value in the other parameter for tensor size.
    # The program automatically calculates what the other unit is supposed to be, in our case, 2.
    # If the unit you put in is not a factor of the original tensor size, eg 9, not a factor of 16
    # The code throws an error.
    print(a)




