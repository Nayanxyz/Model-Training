import torch

# Data we want model to learn

x_data = torch.tensor([1.0, 2.0, 3.0])

y_target = torch.tensor([3.0, 6.0, 9.0])

# The Weight (The matrix)

w = torch.tensor([5.0], requires_grad=True)                      # random weight,  requires_grad=True is the most important,
                                                                    # it tells pyTorch's calculus engine to track this variable


