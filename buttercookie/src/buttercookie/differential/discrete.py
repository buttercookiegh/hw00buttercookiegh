def diff(t, x):
    """
    Compute the discrete derivative of a timeseries.
    
    The discrete derivative nu(t) is given by:
    nu(t) = (x(t_k) - x(t_{k-1})) / (t_k - t_{k-1})
    """
    # Check if the input arrays have equal length
    if len(t) != len(x):
        raise ValueError("Time and signal arrays must have the same length")
    
    n = len(t)
    nu = [0.0] * n  # Initialize derivative array
    
    # Compute discrete derivative for each point starting from index 1
    for k in range(1, n):
        dt = t[k] - t[k-1]  # Time step
        if dt == 0:
            raise ValueError(f"Time step cannot be zero at index {k}")
        # Discrete derivative: (x(t_k) - x(t_{k-1})) / (t_k - t_{k-1})
        nu[k] = (x[k] - x[k-1]) / dt
    
    return nu
