# levy.py
import numpy as np
import cupy as cp
from cupyx.scipy.special import gamma

def levy(n, m, beta):
    """
    CPU version of Levy flight implementation
    Parameters:
    n : Number of steps
    m : Number of Dimensions
    beta : Power law index (1 < beta < 2)
    """
    num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    z = u / (np.abs(v) ** (1 / beta))
    return z

def levy_gpu(n, m, beta):
    """
    GPU version of Levy flight implementation
    Parameters:
    n : Number of steps
    m : Number of Dimensions
    beta : Power law index (1 < beta < 2)
    """
    try:
        with cp.cuda.Device(0):
            num = gamma(1 + beta) * cp.sin(cp.pi * beta / 2)
            den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
            sigma_u = (num / den) ** (1 / beta)

            u = cp.random.normal(0, sigma_u, (n, m))
            v = cp.random.normal(0, 1, (n, m))

            z = u / (cp.abs(v) ** (1 / beta))
            return cp.asnumpy(z)
    except:
        print("GPU not available, falling back to CPU implementation")
        return levy(n, m, beta)