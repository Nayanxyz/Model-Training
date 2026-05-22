import torch

# Data we want model to learn

x_data = torch.tensor([1.0, 2.0, 3.0])

y_target = torch.tensor([3.0, 6.0, 9.0])

# The Weight (The matrix)

w = torch.tensor([5.0], requires_grad=True)                      # random weight,  requires_grad=True is the most important,
                                                                    # it tells pyTorch's calculus engine to track this variable

#Hyperparameter

learning_rate = 0.01

print(f"Initial random weight: {w.item():.4f}")


# The Training Loop
# step 1 :- y = w * x
for epoch in range(100):
    y_pred = w * x_data

    # step 1:- The Loss Function: Measuring the error

    loss = torch.mean((y_pred - y_target) ** 2)

    # step 3:- The Calculus
    loss.backward()

    # step 4 :- Updating the Weight

    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Weight is now {w.item():.4f}, Loss is {loss.item():.4f}")

print(f"\n Final trained weight : {w.item()}")
