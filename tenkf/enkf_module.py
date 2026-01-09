import numpy as np
from lorenz96_model import Lorenz96, RK4, h, n, F, dt

def Dh(x, config=None):
    if config is not None:
        from lorenz96_model import get_observation_operator_matrix
        return get_observation_operator_matrix(config)
    n_state = x.shape[0]
    m = 9
    H = np.zeros((m, n_state))
    di = int(n_state/m)
    for i in range(m):
        H[i, (i+1)*di-1] = 1
    return H

def EnKF(xbi, yo, ObsOp, JObsOp, R, config=None):
    n_state, N = xbi.shape
    m = yo.shape[0]
    
    xb = np.mean(xbi, 1)
    Dh_mat = JObsOp(xb, config) if config is not None else JObsOp(xb)
    
    B = (1/(N-1)) * (xbi - xb.reshape(-1,1)) @ (xbi - xb.reshape(-1,1)).T
    eps = 1e-8 * (R[0,0] if R.ndim > 1 else R) + 1e-10
    R_stable = R + eps * np.eye(R.shape[0])
    D = Dh_mat @ B @ Dh_mat.T + R_stable
    K = B @ Dh_mat.T @ np.linalg.inv(D)
    
    yoi = np.zeros([m, N])
    xai = np.zeros([n_state, N])
    
    for i in range(N):
        yoi[:, i] = yo + np.random.multivariate_normal(np.zeros(m), R)
        obs_pred = ObsOp(xbi[:, i], config) if config is not None else ObsOp(xbi[:, i])
        xai[:, i] = xbi[:, i] + K @ (yoi[:, i] - obs_pred)
    
    return xai

def run_experiment(x0b, yo, ind_m, nt, nt_m, N, Q, R, sig_b, verbose: bool = False, config=None):
    xai = np.zeros([n, N])
    for i in range(N):
        xai[:, i] = x0b + np.random.multivariate_normal(np.zeros(n), sig_b**2 * np.eye(n))
    
    xa = np.zeros([n, nt+1])
    xa[:, 0] = np.mean(xai, 1)
    km = 0
    
    for k in range(nt):
        for i in range(N):
            xai[:, i] = RK4(Lorenz96, xai[:, i], dt, F) + np.random.multivariate_normal(np.zeros(n), Q)
        
        xa[:, k+1] = np.mean(xai, 1)
        
        if (km < nt_m) and (k+1 == ind_m[km]):
            ObsOp = (lambda x, cfg=None: h(x, config)) if config is not None else (lambda x, cfg=None: h(x))
            xai = EnKF(xai, yo[:, km], ObsOp, Dh, R, config)
            xa[:, k+1] = np.mean(xai, 1)
            km = km + 1

    return xa, xai


def run_complete_enkf_experiment(config, verbose: bool = False):
    from data_generator import generate_data, generate_params
    
    xTrue, yo, _, _, _ = generate_data(config, verbose=verbose)
    
    Q, R, B = generate_params(config)
    
    x0b = xTrue[:, 0] + np.random.normal(0, config.sig_b, [n,])
    
    xa, xai = run_experiment(x0b, yo, config.ind_m, config.nt, config.nt_m,
                            config.N, Q, R, config.sig_b, verbose=verbose, config=config)
    
    return xTrue, xa, xai, yo

