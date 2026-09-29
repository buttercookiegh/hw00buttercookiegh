import secrets
import math

def uniform(a=0.0, b=1.0):
    u = secrets.randbits(53) / (1 << 53)
    return a + (b - a) * u

def exponentialdist(lam):
    y = uniform()
    return -(1.0 / lam) * math.log(y)

def poissondist(lam):
    u = uniform()
    k, p, F = 0, math.exp(-lam), math.exp(-lam)
    while u > F:
        k += 1
        p *= lam / k
        F += p
    return k
