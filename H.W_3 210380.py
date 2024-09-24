import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x1, x2):
    return (x1**2 + x2 - 11)**2 + (x2**2 + x1  - 7)**2

# Define the gradient of the function
def grad_f(x1, x2):
    df_dx1 = 4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7)
    df_dx2 = 2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
    return np.array([df_dx1, df_dx2])

# Line search to find the optimal step size
def line_search(x, direction):
    alpha = 1.0
    c = 1e-4
    rho = 0.9
    while f(x[0] + alpha * direction[0], x[1] + alpha * direction[1]) > f(x[0], x[1]) + c * alpha * np.dot(grad_f(x[0], x[1]), direction):
        alpha *= rho
    return alpha


# Steepest descent method
def steepest_descent(x0, tol=1e-3,max_iter=1000):
    x = np.array(x0)
    path = [x]
    while np.linalg.norm(grad_f(x[0], x[1])) > tol:
        direction = -grad_f(x[0], x[1])
        alpha = line_search(x, direction)
        print(f"Step size: {alpha}")  # Print the step size at each iteration
        x = x + alpha * direction
        path.append(x)
    return np.array(path)


# Initial point
x0 = [0, 0]

# Perform steepest descent
path = steepest_descent(x0)

# Plotting the function as a surface and contours
x1 = np.linspace(-6, 6, 400)
x2 = np.linspace(-6, 6, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

fig = plt.figure(figsize=(12, 6))

# Surface plot
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Surface Plot')

# Contour plot with path of convergence
ax2 = fig.add_subplot(122)
ax2.contour(X1, X2, Z, levels=50, cmap='viridis')
path = np.array(path)
ax2.plot(path[:, 0], path[:, 1], 'ro-', markersize=5, label='Path of Convergence')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot')
ax2.legend()

plt.show()
