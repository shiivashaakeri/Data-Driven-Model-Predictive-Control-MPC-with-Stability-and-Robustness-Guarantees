import numpy as np

def henkel_r(w, t1, t2, r):
    """
    Constructs a Hankel matrix from the input vector w.

    Parameters:
    - w: Input vector.
    - t1, t2, r: Parameters defining the size and structure of the Hankel matrix.

    Returns:
    - H: The Hankel matrix.
    """
   # Create the extended vector
    x = np.concatenate([w[:r*(t1+t2-1)], np.flipud(w[:r*(t1+t2-r)])])

    # Create the Hankel subscripts
    ij = (np.arange(1, r*t1+1)[:, None] + np.arange(0, r*t2, r)) - 1

    # Actual data
    H = x[ij] 

    # Preserve shape for a single row
    if H.shape[0] == 1:
        H = H.T

    return H
