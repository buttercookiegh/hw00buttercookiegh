import secrets
import math

def uniform(a=0.0, b=1.0):
    """Cryptographically secure uniform sample."""
    u = secrets.randbits(53) / (1 << 53)
    return a + (b - a) * u

def exponentialdist(lam):
    """Generate exponential random sample using inverse transform."""
    y = uniform(0.0, 1.0)
    return -(1.0 / lam) * math.log(y)

def poissondist(lam):
    """Generate Poisson random sample using inverse transform."""
    u = uniform(0.0, 1.0)
    k = 0
    p = math.exp(-lam)
    F = p
    while u > F:
        k += 1
        p *= lam / k
        F += p
    return k
