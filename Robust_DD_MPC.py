import numpy as np
import control
import cvxpy as cp
import scipy.io
from MDL_sim_prestab import MDL_sim_prestab
from henkel_r import henkel_r
import matplotlib.pyplot as plt

import numpy as np



# Clearing variables is not typically done in Python scripts as it is in MATLAB.
# Python variables are local to their scope (usually a function).
# You can, however, reset variables or redefine them as needed.

# Set options
robust = True
TEC = True
tol_opt = 1e-4
# In Python, optimization settings can depend on the specific library used.
# Here, we are using cvxpy for optimization purposes.
opt_settings = {'OptimalityTolerance': 1e-9,
                'MaxIterations': 20000,
                'ConstraintTolerance': 1e-9}

# Create data
n = 4  # True system order
nu = 4  # Estimated system order
M = 10   # Number of consecutive applications of optimal input (multi-step)
noise_max = 0.002  # Measurement noise on y
N = 400  # Length of data used for prediction
L_true = 30  # Actual prediction horizon (without initial conditions)
L = L_true + nu  # Full prediction horizon (including initial conditions)
T = 600  # "Closed-loop horizon" (simulation length)
m = 2; p = 2  # Number of inputs and outputs


data = scipy.io.loadmat('tank_sys2.mat')

# Extract the system matrices
A_d = data['A_d']
B_d = data['B_d']
C = data['C']
D = data['D']
T_s = data['T_s'].item()  # Assuming T_s is a scalar

# Create the state-space system
sys = control.ss(A_d, B_d, C, D, T_s)

# Initial I/O trajectories for simulation to obtain some data
ui = np.random.rand(m, n)  # initial input
xi0 = np.random.rand(n, 1)
xi = np.zeros((n, n))
xi[:, 0] = xi0.squeeze()

for i in range(n-1):
    xi[:, i+1] = sys.A @ xi[:, i] + sys.B @ ui[:, i]

yi = sys.C @ xi + sys.D @ ui
K = np.eye(m) # pre-stabilizing controller

u, x, y = MDL_sim_prestab(sys, ui, yi, K, noise_max, 0, N)
# Terminal constraints
u_term = np.ones((m, 1))
y_term = sys.C @ np.linalg.inv(np.eye(n) - sys.A) @ sys.B @ u_term


# Set up MPC
# Cost matrices
R = 1e-4 * np.eye(m)  # input weighting
Q = 3 * np.eye(p)  # output weighting
S = np.zeros((m, p))

# Constructing the Pi matrix
Pi = np.block([
    [np.kron(np.eye(L), R), np.kron(np.eye(L), S)],
    [np.kron(np.eye(L), S.T), np.kron(np.eye(L), Q)]
])

# Cost for QP
if robust:
    cost_sigma = 1e3
    cost_alpha = 1e-1
    H = 2 * np.block([
        [cost_alpha * np.eye(N-L+1), np.zeros((N-L+1, (m+p)*L)), np.zeros((N-L+1, p*L))],
        [np.zeros(((m+p)*L, N-L+1)), Pi, np.zeros(((m+p)*L, p*L))],
        [np.zeros((p*L, N-L+1 + (m+p)*L)), cost_sigma * np.eye(p*L)]
    ])
    f = np.concatenate([
        np.zeros((N-L+1, 1)),
        -2 * np.kron(np.eye(L), R) @ np.tile(u_term, (L, 1)),
        -2 * np.kron(np.eye(L), Q) @ np.tile(y_term, (L, 1)),
        np.zeros((p*L, 1))
    ])
else:
    cost_alpha = 0
    H = 2 * np.block([
        [cost_alpha * np.eye(N-L+1), np.zeros((N-L+1, (m+p)*L))],
        [np.zeros(((m+p)*L, N-L+1)), Pi]
    ])
    f = np.concatenate([
        np.zeros((N-L+1, 1)),
        -2 * np.kron(np.eye(L), R) @ np.tile(u_term, (L, 1)),
        -2 * np.kron(np.eye(L), Q) @ np.tile(y_term, (L, 1))
    ])



# Inequality constraints

u_max = np.inf * np.ones((m, 1))
u_min = -np.inf * np.ones((m, 1))
y_max = np.inf * np.ones((p, 1))
y_min = -np.inf * np.ones((p, 1))

if robust:
    sigma_max = np.inf * np.ones((p, 1))
    sigma_min = -sigma_max
    ub = np.concatenate([
        np.full((N-L+1, 1), np.inf),
        np.tile(u_max, (L, 1)),
        np.tile(y_max, (L, 1)),
        np.tile(sigma_max, (L, 1))
    ])
    lb = np.concatenate([
        np.full((N-L+1, 1), -np.inf),
        np.tile(u_min, (L, 1)),
        np.tile(y_min, (L, 1)),
        np.tile(sigma_min, (L, 1))
    ])

else:
    ub = np.concatenate([
        np.full((N-L+1, 1), 1),
        np.tile(u_max, (L, 1)),
        np.tile(y_max, (L, 1))
    ])
    lb = np.concatenate([
        np.full((N-L+1, 1), -1),
        np.tile(u_min, (L, 1)),
        np.tile(y_min, (L, 1))
    ])



# Constructing the Hankel matrices
Hu = henkel_r(u.flatten(), L, N-L+1, m)
Hy = henkel_r(y.flatten(), L, N-L+1, p)

# Building the equality constraints
if robust:
    if TEC:  # With terminal constraints
        B = np.block([
    [Hu, -np.eye(m*L), np.zeros((p*L, p*L)), np.zeros((p*L, p*L))],
    [Hy, np.zeros((p*L, m*L)), -np.eye(p*L), -np.eye(p*L)],
    [np.zeros((m*nu, N-L+1)), np.hstack([np.eye(m*nu), np.zeros((m*nu, m*(L-nu)))]), np.zeros((m*nu, p*L)), np.zeros((m*nu, p*L))],
    [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.hstack([np.eye(p*nu), np.zeros((p*nu, p*(L-nu)))]), np.hstack([np.zeros((p*nu, p*nu)), np.zeros((p*nu, p*(L-nu)))])],
    [np.zeros((m*nu, N-L+1)), np.hstack([np.zeros((m*nu, m*(L-nu))), np.eye(m*nu)]), np.zeros((m*nu, p*L)), np.zeros((m*nu, p*L))],
    [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.hstack([np.zeros((p*nu, p*(L-nu))), np.eye(p*nu)]), np.zeros((p*nu, p*L))]
])
        
    else:  # Without terminal constraints
        B = np.block([
            [Hu, -np.eye(m*L), np.zeros(p*L), np.zeros(p*L)],
            [Hy, np.zeros((p*L, m*L)), -np.eye(p*L), -np.eye(p*L)],
            [np.zeros((m*nu, N-L+1)), np.hstack([np.eye(m*nu), np.zeros((m*nu, m*(L-nu)))]), np.zeros((m*nu, p*L)), np.zeros((m*nu, p*L))],
            [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.hstack([np.eye(p*nu), np.zeros((p*nu, p*(L-nu)))]), np.hstack([np.zeros(p*nu), np.zeros((p*nu, p*(L-nu)))])]
        ])
else:
    if TEC:  # With terminal constraints
        B = np.block([
            [Hu, -np.eye(m*L), np.zeros((m*L, p*L))],
            [Hy, np.zeros((p*L, m*L)), -np.eye(p*L)],
            [np.zeros((m*nu, N-L+1)), np.eye(m*nu), np.zeros((m*nu, m*(L-nu))), np.zeros((m*nu, p*L))],
            [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.eye(p*nu), np.zeros((p*nu, p*(L-nu)))]
        ])
    else:  # Without terminal constraints
        B = np.block([
            [Hu, -np.eye(m*L), np.zeros((m*L, p*L))],
            [Hy, np.zeros((p*L, m*L)), -np.eye(p*L)],
            [np.zeros((m*nu, N-L+1)), np.eye(m*nu), np.zeros((m*nu, m*(L-nu)))],
            [np.zeros((p*nu, N-L+1)), np.zeros((p*nu, m*L)), np.eye(p*nu), np.zeros((p*nu, p*(L-nu)))]
        ])


# Initial I/O trajectories
u_init = 0.8 * np.ones((m, nu))  # initial input
x0 = np.array([0.4, 0.4, 0.5, 0.5])  # initial state
x_init = np.zeros((n, nu))
x_init[:, 0] = x0

for i in range(nu-1):
    x_init[:, i+1] = sys.A @ x_init[:, i] + sys.B @ u_init[:, i]

y_init = sys.C @ x_init + sys.D @ u_init

# Closed-loop storage variables
u_cl = np.zeros((m, T))
u_cl[:, :nu] = u_init  # Set initial input

y_cl = np.zeros((p, T))
y_cl[:, :nu] = y_init  # Set initial output

y_cl_noise = np.copy(y_cl)  # Copy of y_cl for noisy output

x_cl = np.zeros((n, T))
x_cl[:, 0] = x0  # Set initial state

# Simulate first nu steps
for j in range(nu):
    x_cl[:, j+1] = sys.A @ x_cl[:, j] + sys.B @ u_cl[:, j]

# Open-loop storage variables
u_ol = np.zeros((m * L, T))
y_ol = np.zeros((p * L, T))
sigma_ol = np.zeros((p * L, T))
alpha_ol = np.zeros((N - L + 1, T))
u_init_store = np.zeros((m * nu, T))
y_init_store = np.zeros((p * nu, T))

# Candidate solution storage variables
u_cand = np.copy(u_ol)
y_cand = np.copy(y_ol)
alpha_cand = np.copy(alpha_ol)
fval_cand = np.zeros((1, T))

if robust:
    sol_store = np.zeros(((m + 2 * p) * L + N - L + 1, T))
else:
    sol_store = np.zeros(((m + p) * L + N - L + 1, T))

sol_cand = np.copy(sol_store)
fval = np.zeros(T)

# MPC Loop
for j in range(nu + 1, T, M):
    print(j)
      # To display the current iteration
    # Update equality constraints
    if TEC:
        c = np.concatenate([
            np.zeros(((m + p) * L, 1)), 
            u_init.flatten()[:, np.newaxis],  # Convert to 2D column vector
            y_init.flatten()[:, np.newaxis],  # Convert to 2D column vector
            np.tile(u_term, (nu, 1)), 
            np.tile(y_term, (nu, 1))
        ], axis=0)
    else:
        c = np.concatenate([
            np.zeros(((m + p) * L, 1)), 
            u_init.flatten()[:, np.newaxis],  # Convert to 2D column vector
            y_init.flatten()[:, np.newaxis]  # Convert to 2D column vector
        ], axis=0)

    # Define and solve the QP problem using cvxpy or another QP solver
    x = cp.Variable((H.shape[0], 1))
    objective = cp.Minimize(0.5 * cp.quad_form(x, H) + f.T @ x)
    constraints = [B @ x == c,lb <= x, x <= ub] 
    # Solver options for OSQP

    prob = cp.Problem(objective, constraints)
    osqp_options = {
    'eps_abs': 1e-5,  # Absolute tolerance
    'eps_rel': 1e-5,  # Relative tolerance
    'max_iter': 50000  # Maximum iterations
}

# Solve the problem with adjusted settings
    result = prob.solve(solver=cp.OSQP, **osqp_options)

    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError("Optimization problem not solved exactly..")
    if prob.status in ["optimal", "optimal_inaccurate"]:

        sol = x.value.flatten()
        fval[j] = prob.value + np.dot(np.tile(y_term, (L, 1)).T, np.kron(np.eye(L), Q) @ np.tile(y_term, (L, 1))) + np.dot(np.tile(u_term, (L, 1)).T, np.kron(np.eye(L), R) @ np.tile(u_term, (L, 1)))
        sol_store[:, j] = sol
        alpha_ol[:, j] = sol[:N-L+1]
        u_ol[:, j] = sol[N-L+1:N-L+1+m*L]
        y_ol[:, j] = sol[N-L+1+m*L:N-L+1+(m+p)*L]
        if robust:
            sigma_ol[:, j] = sol[N-L+1+(m+p)*L:N-L+1+(m+2*p)*L]

        u_init_store[:, j-nu] = u_init.flatten()
        y_init_store[:, j-nu] = y_init.flatten()

    # Simulate closed loop
    for k in range(j, min(j + M, T - 1)):
        u_cl[:, k] = u_ol[m*n + (k - j) * m : m*n + m + (k - j) * m, j]
    
    # Update x_cl only if k + 1 is within bounds
        if k + 1 < T:
            x_cl[:, k + 1] = sys.A @ x_cl[:, k] + sys.B @ u_cl[:, k]

        y_cl[:, k] = sys.C @ x_cl[:, k] + sys.D @ u_cl[:, k]
        y_cl_noise[:, k] = y_cl[:, k] * (1 + noise_max * (-1 + 2 * np.random.rand(p, 1))).flatten()

    # Set new initial conditions for the next iteration
        u_init = np.hstack([u_init[:, 1:], u_cl[:, k:k+1]])
        y_init = np.hstack([y_init[:, 1:], y_cl_noise[:, k:k+1].reshape(p, 1)])


        print(u_init)
plt.figure()

# Subplot for u_1
plt.subplot(2, 2, 1)
plt.plot(range(1, T + 1), u_cl[0, :], label='u_1')
plt.plot(range(1, T + 1), [u_term[0]] * T, label='u_{1,eq}')
plt.legend()

# Subplot for u_2
plt.subplot(2, 2, 2)
plt.plot(range(1, T + 1), u_cl[1, :], label='u_2')
plt.plot(range(1, T + 1), [u_term[1]] * T, label='u_{2,eq}')
plt.legend()

# Subplot for y_1
plt.subplot(2, 2, 3)
plt.plot(range(1, T + 1), y_cl[0, :], label='y_1')
plt.plot(range(1, T + 1), [y_term[0]] * T, label='y_{1,eq}')
plt.legend()

# Subplot for y_2
plt.subplot(2, 2, 4)
plt.plot(range(1, T + 1), y_cl[1, :], label='y_2')
plt.plot(range(1, T + 1), [y_term[1]] * T, label='y_{2,eq}')
plt.legend()

# Show the plot
plt.show()
