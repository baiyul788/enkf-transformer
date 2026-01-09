import os
import numpy as np
from lorenz96_model import Lorenz96, RK4, h, n, F, dt
from enkf_module import run_experiment as run_enkf_experiment

def generate_data(config, verbose: bool = True):
    nt = config.nt
    nt_m = config.nt_m
    ind_m = config.ind_m
    
    x0 = F * np.ones(n)
    x0[19] = x0[19] + 0.01
    x0True = x0
    ntl = int(config.spinup_time/dt)
    for k in range(ntl):
        x0True = RK4(Lorenz96, x0True, dt, F)
    
    xTrue = np.zeros([n, nt+1])
    xTrue[:, 0] = x0True
    km = 0
    yo = np.zeros([config.s, nt_m])
    
    for k in range(nt):
        xTrue[:, k+1] = RK4(Lorenz96, xTrue[:, k], dt, F)
        if (km < nt_m) and (k+1 == ind_m[km]):
            yo[:, km] = h(xTrue[:, k+1], config) + np.random.normal(0, config.sig_m, [config.s,])
            km = km + 1
    
    return xTrue, yo, ind_m, nt, nt_m

def generate_params(config):
    Q = config.sig_p**2 * np.eye(n)
    R = config.sig_m**2 * np.eye(config.s)
    B = config.sig_b**2 * np.eye(n)
    return Q, R, B

def generate_enkf_residual_training_data(n_samples=100, tm=5, dt_m=0.2, k=5,
                                         N=100, sig_p=0.02, sig_b=1.0, sig_m=0.1,
                                         spinup_time=20, seed=1, *, verbose=True, progress_every=10,
                                         enable_cache: bool = False, cache_dir: str = "results"):
    if enable_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"residual_data_ns{n_samples}_tm{tm}_dtm{dt_m}_k{k}_N{N}_sp{sig_p}_sb{sig_b}_sm{sig_m}_seed{seed}.npz"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["train_input"], data["train_target"], data["val_input"], data["val_target"]

    dt_loc = dt
    nt = int(tm / dt_loc)
    nt_m = int(tm / dt_m)
    ind_m = (np.linspace(int(dt_m/dt_loc), int(tm/dt_loc), nt_m)).astype(int)

    T_ds = nt // k + 1
    inputs = np.zeros((n_samples, T_ds, n))
    targets = np.zeros((n_samples, T_ds, n))

    for sample in range(n_samples):
        from config.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            tm=tm, dt_m=dt_m, sig_m=sig_m, spinup_time=spinup_time,
            N=N, sig_p=sig_p, sig_b=sig_b, seed=seed + sample
        )
        from enkf_module import run_complete_enkf_experiment
        xTrue, xa, _, yo = run_complete_enkf_experiment(config, verbose=False)

        xa_ds = xa[:, ::k].T
        xTrue_ds = xTrue[:, ::k].T

        inputs[sample] = xa_ds
        targets[sample] = xTrue_ds - xa_ds

    val_count = int(n_samples * 0.2)
    train_input = inputs[val_count:]
    train_target = targets[val_count:]
    val_input = inputs[:val_count]
    val_target = targets[:val_count]

    if enable_cache:
        np.savez(cache_path,
                 train_input=train_input, train_target=train_target,
                 val_input=val_input, val_target=val_target)

    return train_input, train_target, val_input, val_target


def generate_eakf_residual_training_data(n_samples=100, tm=5, dt_m=0.2, k=5,
                                         N=100, sig_p=0.02, sig_b=1.0, sig_m=0.1,
                                         spinup_time=20, seed=1, *, verbose=True, progress_every=10,
                                         enable_cache: bool = False, cache_dir: str = "results"):
    if enable_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"residual_data_eakf_ns{n_samples}_tm{tm}_dtm{dt_m}_k{k}_N{N}_sp{sig_p}_sb{sig_b}_sm{sig_m}_seed{seed}.npz"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["train_input"], data["train_target"], data["val_input"], data["val_target"]

    dt_loc = dt
    nt = int(tm / dt_loc)
    nt_m = int(tm / dt_m)
    ind_m = (np.linspace(int(dt_m/dt_loc), int(tm/dt_loc), nt_m)).astype(int)

    T_ds = nt // k + 1
    inputs = np.zeros((n_samples, T_ds, n))
    targets = np.zeros((n_samples, T_ds, n))

    for sample in range(n_samples):
        from config.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            tm=tm, dt_m=dt_m, sig_m=sig_m, spinup_time=spinup_time,
            N=N, sig_p=sig_p, sig_b=sig_b, seed=seed + sample
        )

        from eakf_module import run_complete_eakf_experiment
        xTrue, xa, _, yo = run_complete_eakf_experiment(config, verbose=False)

        xa_ds = xa[:, ::k].T
        xTrue_ds = xTrue[:, ::k].T

        inputs[sample] = xa_ds
        targets[sample] = xTrue_ds - xa_ds

    val_count = int(n_samples * 0.2)
    train_input = inputs[val_count:]
    train_target = targets[val_count:]
    val_input = inputs[:val_count]
    val_target = targets[:val_count]

    if enable_cache:
        np.savez(cache_path,
                 train_input=train_input, train_target=train_target,
                 val_input=val_input, val_target=val_target)

    return train_input, train_target, val_input, val_target