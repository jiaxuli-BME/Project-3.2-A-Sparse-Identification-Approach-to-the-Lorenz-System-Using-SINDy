import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lorenz system dynamics
def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    dxdt = [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
    ]
    return dxdt

# Parameters
sigma, beta, rho = 10, 8/3, 28
n = 3
x0 = [-8, 8, 27]
t_span = (0, 50)
t_eval = np.arange(0.01, 50, 0.01)

# Solve Lorenz system
sol = solve_ivp(lorenz, t_span, x0, args=(sigma, beta, rho), t_eval=t_eval, rtol=1e-12, atol=1e-12*np.ones(n))
x = sol.y.T

# Compute Derivative (dx/dt)
dx = np.array([lorenz(0, x[i, :], sigma, beta, rho) for i in range(len(x))])
dx += np.random.randn(*dx.shape) * 1e-3  # Add small noise

# Build the SINDy model
poly_order = 3  # Up to third-order polynomial terms
lambda_ = 0.025  # Sparsification knob

# Create a SINDy model
model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=poly_order),
                 optimizer=ps.STLSQ(threshold=lambda_))

# Fit the model
model.fit(x, t=t_eval[1] - t_eval[0])
model.print()

# Plot original data and SINDy model prediction
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
labels = ['x', 'y', 'z']
for i in range(3):
    ax[i].plot(t_eval, x[:, i], label=f'{labels[i]} (original)')
    x_dot_pred = model.predict(x)[:, i]
    ax[i].plot(t_eval, x_dot_pred, '--', label=f'{labels[i]} (predicted)')
    ax[i].set_ylabel(labels[i])
    ax[i].legend()
plt.xlabel('Time')
plt.show()
