import numpy as np
from lorenz96_model import n, h, Lorenz96, RK4, dt, F

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

def EAKF(xbi, yo, ObsOp, JObsOp, R, config=None):
    n, N = xbi.shape
    m = yo.shape[0]
    
    xb = np.mean(xbi, axis=1)
    
    H = JObsOp(xb, config) if config is not None else JObsOp(xb)
    
    A = xbi - xb.reshape(-1, 1)
    
    Y = H @ A
    
    yb = ObsOp(xb, config) if config is not None else ObsOp(xb)
    
    P_yy = (1/(N-1)) * Y @ Y.T
    
    jitter = 1e-6
    R_eff = R + jitter * np.eye(R.shape[0])
    S = P_yy + R_eff + 1e-6 * np.eye(R.shape[0])
    
    P_xx = (1/(N-1)) * A @ A.T
    
    K = P_xx @ H.T @ np.linalg.inv(S)
    
    xa = xb + K @ (yo - yb)
    
    try:
        R_inv = np.linalg.inv(R_eff)
        Tm = (1.0/(N-1)) * Y.T @ R_inv @ Y
        Tm = 0.5 * (Tm + Tm.T)
        eigvals, eigvecs = np.linalg.eigh(np.eye(N) + Tm)
        threshold = config.eakf_eigenval_threshold if config else 1e-8
        eigvals = np.clip(eigvals, threshold, 1e6)
        inv_sqrt = np.diag(1.0/np.sqrt(eigvals))
        W = eigvecs @ inv_sqrt @ eigvecs.T
        A_adj = A @ W
        A_adj = A_adj - np.mean(A_adj, axis=1, keepdims=True)
    except np.linalg.LinAlgError:
        alpha = config.eakf_fallback_alpha if config else 0.8
        A_adj = alpha * A
    
    xai = np.zeros([n, N])
    for i in range(N):
        xai[:, i] = xa + A_adj[:, i]
    
    return xai


def run_eakf_experiment(x0b, yo, ind_m, nt, nt_m, N, Q, R, sig_b, verbose: bool = True, config=None):
    xai = np.zeros([n, N])
    for i in range(N):
        xai[:, i] = x0b + np.random.multivariate_normal(np.zeros(n), sig_b**2 * np.eye(n))
    
    xa = np.zeros([n, nt+1])
    xa[:, 0] = np.mean(xai, 1)
    
    ObsOp = (lambda x, cfg=None: h(x, config)) if config is not None else (lambda x, cfg=None: h(x))
    JObsOp = (lambda x, cfg=None: Dh(x, config)) if config is not None else (lambda x, cfg=None: Dh(x))

    km = 0
    for k in range(nt):
        for i in range(N):
            xai[:, i] = RK4(Lorenz96, xai[:, i], dt, F) + np.random.multivariate_normal(np.zeros(n), Q)
        
        xa[:, k+1] = np.mean(xai, 1)
        
        if (km < nt_m) and (k+1 == ind_m[km]):
            if hasattr(config, 'inflation_factor') and (config.inflation_factor is not None):
                xb = np.mean(xai, axis=1, keepdims=True)
                A = xai - xb
                xai = xb + np.sqrt(config.inflation_factor) * A

            xai = EAKF(xai, yo[:, km], ObsOp, JObsOp, R, config)
            xa[:, k+1] = np.mean(xai, 1)
            km += 1

    return xa, xai


def run_complete_eakf_experiment(config, verbose: bool = False):
    from data_generator import generate_data, generate_params

    xTrue, yo, _, _, _ = generate_data(config, verbose=verbose)

    Q, R, B = generate_params(config)

    x0b = xTrue[:, 0] + np.random.normal(0, config.sig_b, [n,])

    xa, xai = run_eakf_experiment(x0b, yo, config.ind_m, config.nt, config.nt_m,
                                  config.N, Q, R, config.sig_b, verbose=verbose, config=config)

    return xTrue, xa, xai, yo
