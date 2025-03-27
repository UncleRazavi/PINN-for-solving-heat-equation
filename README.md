# Physics-Informed Neural Network (PINN) for Solving the Heat Equation

## Overview
This Python script implements a Physics-Informed Neural Network (PINN) using PyTorch to solve the one-dimensional heat equation. The network approximates the solution to the partial differential equation (PDE) by minimizing the residual of the PDE along with boundary and initial conditions.

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install torch numpy matplotlib
```

## Model Definition
The `PINN` class defines a fully connected neural network with multiple hidden layers using the Tanh activation function.

### `PINN(nn.Module)`
#### **Constructor: `__init__(self, layers)`**
- **Parameters:**
  - `layers`: A list defining the number of neurons in each layer.
- **Functionality:**
  - Initializes a feedforward neural network with Xavier-normal initialization.
  - Applies a Tanh activation function to hidden layers.

#### **Forward Pass: `forward(self, X)`**
- **Input:** Tensor `X` containing spatial and temporal coordinates.
- **Output:** Predicted solution `u(x,t)`.

## Data Generation
### `generate_data()`
- Generates a grid of spatial (`x`) and temporal (`t`) points in the domain [0,1] × [0,1].
- Converts the data into a PyTorch tensor.

## Loss Function
### `compute_Loss(model, X, alpha=0.05)`
This function calculates the total loss based on:
1. **PDE Residual:** The residual of the heat equation.
2. **Boundary Conditions:** Enforcing `u(0,t) = u(1,t) = 0`.
3. **Initial Condition:** Enforcing `u(x,0) = sin(pi*x)`.

- **Inputs:**
  - `model`: The PINN model.
  - `X`: Training data containing spatial and temporal points.
  - `alpha`: Diffusion coefficient (default: 0.05).
- **Output:**
  - Scalar loss value for training.

## Training Function
### `train_PINN_model(model, X_train, epochs=5000, lr=0.001)`
Trains the PINN using Adam optimizer.

- **Inputs:**
  - `model`: The PINN neural network.
  - `X_train`: Training data points.
  - `epochs`: Number of training iterations.
  - `lr`: Learning rate.
- **Outputs:**
  - Prints the loss every 500 epochs.

## Running the PINN Model
- The model is initialized with the layer structure `[2, 32, 32, 32, 1]`.
- Training data is generated using `generate_data()`.
- The model is trained using `train_PINN_model()`.

## Visualization
- The trained model is used to predict the temperature distribution.
- The results are plotted using `matplotlib.pyplot` to display the solution as a contour plot.

## Example Output
The final visualization shows the predicted temperature `u(x,t)` over space and time using a color map.

---

This implementation provides a deep learning-based approach to solving PDEs using PINNs, allowing flexible and data-driven solutions for physical problems.

