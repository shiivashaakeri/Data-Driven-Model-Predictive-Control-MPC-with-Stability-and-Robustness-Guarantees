import numpy as np
from scipy.linalg import solve, block_diag

def compute_init_cond(sys, u_init, y_init):
    """
    Computes the initial condition for a given system.

    Parameters:
    - sys: A system object with A, B, C, D matrices.
    - u_init: Initial input.
    - y_init: Initial output.

    Returns:
    - x0: The computed initial condition.
    """
    # Get parameters
    n = u_init.shape[1]
    m = u_init.shape[0]
    p = y_init.shape[0]

    # Markov Parameter
    markov = np.zeros((n*p, n*m))
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i > j:
                markov[(i-1)*p:i*p, (j-1)*m:j*m] = sys['C'] @ np.linalg.matrix_power(sys['A'], i-j-1) @ sys['B']
            elif i == j:
                markov[(i-1)*p:i*p, (j-1)*m:j*m] = sys['D']

    # Compute the initial condition
    # Create the observability matrix
    obsv_matrix = np.vstack([sys['C'] @ np.linalg.matrix_power(sys['A'], i) for i in range(n)])

    # Solve for x0
    x0 = solve(obsv_matrix, y_init.flatten() - markov @ u_init.flatten(), assume_a='pos')

    return x0
