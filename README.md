# Physics-Informed Neural Network (PINN) for Solving the Heat Equation

## Overview
This Python script implements a Physics-Informed Neural Network (PINN) using PyTorch to solve the one-dimensional heat equation. The network approximates the solution to the partial differential equation (PDE) by minimizing the residual of the PDE along with boundary and initial conditions.
# Physics-Informed Neural Networks (PINNs) and Their Structure

## Introduction to PINNs

Physics-Informed Neural Networks (PINNs) are a class of deep learning methods that seamlessly integrate the knowledge of physical laws, typically expressed as differential equations, into the training of neural networks. PINNs provide a mesh-free and data-efficient framework to solve forward and inverse problems governed by Partial Differential Equations (PDEs) or Ordinary Differential Equations (ODEs). By embedding physics into the learning architecture, PINNs enable solutions that are consistent with both observed data and underlying scientific principles.

## Mathematical Foundation

Consider a physical problem described by a general nonlinear differential operator $$\(\mathcal{N}\)$$ applied to an unknown function $$\(u(\mathbf{x}, t)\)$$:

$$
\mathcal{N}[u(\mathbf{x}, t)] = 0, \quad (\mathbf{x}, t) \in \Omega \times [0,T],
$$

where $$\(\Omega \subset \mathbb{R}^d\)$$ is the spatial domain and $$\(t\)$$ represents time.  
The problem is usually accompanied by initial and boundary conditions:

$$
\begin{cases}
\mathcal{B}[u(\mathbf{x}, t)] = g(\mathbf{x}, t), & (\mathbf{x}, t) \in \partial \Omega \times [0,T], \\
u(\mathbf{x}, 0) = u_0(\mathbf{x}), & \mathbf{x} \in \Omega,
\end{cases}
$$

where $$\(\mathcal{B}\)$$ is a boundary operator and $$\(g, u_0\)$$ are known boundary and initial data, respectively.

The goal of PINNs is to approximate $$\(u(\mathbf{x}, t)\)$$ by a neural network $$\(\hat{u}(\mathbf{x}, t; \theta)\)$$ parameterized by weights and biases $$\(\theta\)$$.

---

## PINN Architecture

- **Input Layer:**  
  Receives spatial coordinates and time points $$\((\mathbf{x}, t)\)$$.

- **Hidden Layers:**  
  Several fully connected layers with smooth nonlinear activation functions (e.g., hyperbolic tangent \(\tanh\), sine, ReLU). These layers allow the network to model complex nonlinear behaviors inherent in physical phenomena.

- **Output Layer:**  
  Predicts the scalar or vector-valued solution $$\(\hat{u}(\mathbf{x}, t)\)$$.

---

## Incorporating Physics via Loss Functions

A key component differentiating PINNs from traditional neural networks is their loss function, which explicitly enforces the governing physics by minimizing the PDE residuals.

### PDE Residual Loss

Using automatic differentiation, the derivatives required by $$\(\mathcal{N}\)$$ are computed exactly on the neural network output $$\(\hat{u}\)$$. The PDE residual is:

$$
f(\mathbf{x}, t; \theta) = \mathcal{N}[\hat{u}(\mathbf{x}, t; \theta)].
$$

The physics loss term is the mean squared error of these residuals evaluated at a set of collocation points $$\(\{(\mathbf{x}_i, t_i)\}_{i=1}^{N_f}\)$$:

$$
L_{\text{Physics}}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| f(\mathbf{x}_i, t_i; \theta) \right|^2.
$$

### Data Loss

If measurements or exact solution values $$\(\{u_j\}\)$$ at points

$$
\(\{(\mathbf{x}_j, t_j)\}_{j=1}^{N_d}\)
$$ 

are available, a supervised loss term is added:

$$
L_{\text{Data}}(\theta) = \frac{1}{N_d} \sum_{j=1}^{N_d} \left| \hat{u}(\mathbf{x}_j, t_j; \theta) - u_j \right|^2.
$$

### Boundary and Initial Condition Loss

To ensure physical consistency at the domain boundaries and initial time, corresponding loss terms are included:

$$
L_{\text{Boundary}}(\theta) = \frac{1}{N_b} \sum_{k=1}^{N_b} \left| \mathcal{B}[\hat{u}(\mathbf{x}_k, t_k; \theta)] - g(\mathbf{x}_k, t_k) \right|^2,
$$

$$
L_{\text{Initial}}(\theta) = \frac{1}{N_0} \sum_{m=1}^{N_0} \left| \hat{u}(\mathbf{x}_m, 0; \theta) - u_0(\mathbf{x}_m) \right|^2.
$$

### Combined Loss Function

The total loss minimized during training is the weighted sum:

$$
L(\theta) = \lambda_f L_{\text{Physics}}(\theta) + \lambda_d L_{\text{Data}}(\theta) + \lambda_b L_{\text{Boundary}}(\theta) + \lambda_0 L_{\text{Initial}}(\theta),
$$

where $$\(\lambda_{\cdot}\)$$ denote user-defined weights balancing the contribution of each term.

---

## Theoretical Intuition and Convergence

Under regularity assumptions on $$\(u\)$$ and $$\(\mathcal{N}\)$$, the PINN framework ensures consistency because minimizing the residual loss enforces the network to approximate $$\(u\)$$ solving the PDE. Automatic differentiation yields exact gradient computations of the network output, enabling precise evaluation of derivatives without finite difference approximations. This leads to mesh-free, high-accuracy approximations with convergence properties linked to universal approximation theorems for neural networks and the consistency of PDE residual minimization.

---

## Illustrative Case Study: 1D Heat Equation

Consider the heat equation describing heat diffusion in a one-dimensional rod:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0,L], \ t \in [0,T],
$$

with thermal diffusivity constant \(\alpha > 0\).

- **Physics Residual:**

$$
f(x,t; \theta) = \frac{\partial \hat{u}}{\partial t}(x,t;\theta) - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}(x,t;\theta).
$$

- **Boundary and Initial conditions** must also be specified, e.g., Dirichlet boundaries:

$$
u(0,t) = u_L(t), \quad u(L,t) = u_R(t), \quad u(x,0) = u_0(x).
$$

The PINN seeks \(\hat{u}(x,t;\theta)\) that minimizes the combined loss from these components.

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

