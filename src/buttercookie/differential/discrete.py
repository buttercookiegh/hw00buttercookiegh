def diff(t, x):
    if len(t) != len(x):
        raise ValueError("Time and signal arrays must have the same length")
    
    n = len(t)
    nu = [0.0] * n
    
    for k in range(1, n):
        dt = t[k] - t[k-1]
        if dt == 0:
            raise ValueError(f"Time step cannot be zero at index {k}")
        nu[k] = (x[k] - x[k-1]) / dt
    
    return nu
