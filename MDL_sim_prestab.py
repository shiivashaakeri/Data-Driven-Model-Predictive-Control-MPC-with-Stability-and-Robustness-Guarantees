import numpy as np
def observability_matrix(A, C):
    """
    Compute the observability matrix for a system with matrices A and C.

    Parameters:
    - A: System state matrix
    - C: Output matrix

    Returns:
    - Obs: Observability matrix
    """
    n = A.shape[0]
    Obs = C
    for _ in range(1, n):
        Obs = np.vstack((Obs, C @ np.linalg.matrix_power(A, _)))
    return Obs

def MDL_sim_prestab(sys, u_init, y_init, K, noise_max, MULT_NOISE, N):
    
    # Parameters
    n = y_init.shape[1]
    m = u_init.shape[0]
    p = y_init.shape[0]

    # Constructing the Markov parameters
    markov = np.zeros((n*p, n*m))
    for i in range(n):
        for j in range(n):
            if i > j:
                markov[i*p:(i+1)*p, j*m:(j+1)*m] = sys.C @ np.linalg.matrix_power(sys.A, i-j-1) @ sys.B
            elif i == j:
                markov[i*p:(i+1)*p, j*m:(j+1)*m] = sys.D

    # Initial state
    Obs = observability_matrix(sys.A, sys.C)
    x0 = np.linalg.pinv(Obs) @ (y_init.flatten() - markov @ u_init.flatten())

    # Check if x0 has the correct shape
    if x0.shape[0] != n:
        raise ValueError(f"Initial state x0 must be of length {n}")
    

    # Model-based simulation
    x = np.zeros((n, N))
    x[:, 0] = x0
    u = 2 * np.random.rand(m, N) - 1
    u[:, :n] = u_init
    y = np.zeros((p, N))
    y[:, :n] = y_init
    eps = np.random.rand(p, N)

    # Simulation loop
    for i in range(N-1):
        if i >= n:
            u[:, i] = u[:, i] + K @ y[:, i]
        x[:, i+1] = sys.A @ x[:, i] + sys.B @ u[:, i]
        if MULT_NOISE:
            y[:, i+1] = (sys.C @ x[:, i+1] + sys.D @ u[:, i+1]) * (1 + noise_max * (-1 + 2 * eps[:, i+1]))
        else:
            y[:, i+1] = sys.C @ x[:, i+1] + sys.D @ u[:, i+1] + noise_max * (-1 + 2 * eps[:, i+1])

    # Flatten u and y for output
    u = u.flatten()
    y = y.flatten()

    return u, x, y
