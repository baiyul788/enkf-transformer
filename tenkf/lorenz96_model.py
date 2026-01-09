import numpy as np


def Lorenz96(state, *args):
    x = state
    F = args[0]
    n = len(x)
    f = np.zeros(n)
    
    f[0] = (x[1] - x[n-2]) * x[n-1] - x[0]
    f[1] = (x[2] - x[n-1]) * x[0] - x[1]
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    
    for i in range(2, n-1):
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    
    f = f + F
    return f


def RK4(rhs, state, dt, *args):
    CLIP_MIN, CLIP_MAX = -50.0, 50.0

    def _clip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=CLIP_MAX, neginf=CLIP_MIN)
        return np.clip(x, CLIP_MIN, CLIP_MAX)

    s1 = _clip(state)
    k1 = rhs(s1, *args)

    s2 = _clip(s1 + k1 * dt / 2.0)
    k2 = rhs(s2, *args)

    s3 = _clip(s1 + k2 * dt / 2.0)
    k3 = rhs(s3, *args)

    s4 = _clip(s1 + k3 * dt)
    k4 = rhs(s4, *args)

    new_state = s1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    new_state = _clip(new_state)
    return new_state


def h(x, config=None):
    if config is not None and hasattr(config, 'obs_indices'):
        return x[config.obs_indices]
    n = x.shape[0]
    m = 9
    di = int(n/m)
    obs_indices = [(i+1)*di-1 for i in range(m)]
    return x[obs_indices]


def get_observation_operator_matrix(config):
    H = np.zeros((config.s, config.n))
    for i, idx in enumerate(config.obs_indices):
        H[i, idx] = 1
    return H


n = 36
F = 8
dt = 0.01

dt_m = 0.2
tm_m = 20
