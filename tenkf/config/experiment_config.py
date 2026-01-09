import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ExperimentConfig:
    n: int = 36
    F: float = 8.0
    dt: float = 0.01
    
    dt_m: float = 0.2
    sig_m: float = 0.1
    s: int = 9
    obs_strategy: str = "uniform"
    custom_obs_indices: Optional[List[int]] = None
    
    N: int = 100
    sig_p: float = 0.02
    sig_b: float = 1.0

    use_standard_eakf: bool = False
    eakf_eigenval_threshold: float = 1e-10
    eakf_fallback_alpha: float = 0.8
    inflation_factor: float = 1.00

    
    tm: float = 5.0
    k: int = 5
    seed: int = 30
    spinup_time: float = 20.0
    
    n_samples: int = 100
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.0005
    patience: int = 20
    
    d_model: int = 36
    num_heads: int = 6
    num_layers: int = 2
    d_ff: int = 144
    dropout: float = 0.1
    max_len: int = 5000
    
    nt: int = field(init=False)
    nt_m: int = field(init=False)
    ind_m: List[int] = field(init=False)
    obs_indices: List[int] = field(init=False)
    
    def __post_init__(self):
        self.nt = int(self.tm / self.dt)
        self.nt_m = int(self.tm / self.dt_m)
        self.ind_m = (np.linspace(
            int(self.dt_m / self.dt), 
            int(self.tm / self.dt), 
            self.nt_m
        )).astype(int).tolist()
        
        if self.custom_obs_indices is not None:
            self.obs_indices = self.custom_obs_indices
            self.s = len(self.custom_obs_indices)
            if self.s > self.n:
                raise ValueError(f"Custom observation dimension s={self.s} cannot exceed state dimension n={self.n}")
        elif self.obs_strategy == "uniform":
            if self.s > self.n:
                raise ValueError(f"Observation dimension s={self.s} cannot exceed state dimension n={self.n}")
            if self.n % self.s == 0:
                di = self.n // self.s
                self.obs_indices = [(i+1)*di-1 for i in range(self.s)]
            else:
                obs_indices_raw = np.linspace(0, self.n-1, self.s).round().astype(int)
                self.obs_indices = np.unique(obs_indices_raw).tolist()
                if len(self.obs_indices) < self.s:
                    all_indices = set(range(self.n))
                    selected_indices = set(self.obs_indices)
                    remaining_indices = list(all_indices - selected_indices)
                    need_more = self.s - len(self.obs_indices)
                    self.obs_indices.extend(remaining_indices[:need_more])
                    self.obs_indices.sort()
        else:
            raise ValueError(f"Unsupported observation strategy: {self.obs_strategy}")
    
    def get_model_config(self):
        return {
            'input_dim': self.n,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'max_len': self.max_len,
            'dropout': self.dropout
        }
    
    def get_training_config(self):
        return {
            'epochtime': self.epochs,
            'learning_rate': self.lr,
            'batch_size': self.batch_size,
            'patience': self.patience
        }
    
    def get_data_config(self):
        return {
            'n_samples': self.n_samples,
            'tm': self.tm,
            'dt_m': self.dt_m,
            'k': self.k,
            'N': self.N,
            'sig_p': self.sig_p,
            'sig_b': self.sig_b,
            'sig_m': self.sig_m,
            'spinup_time': self.spinup_time,
            'seed': self.seed
        }
    
    def get_enkf_config(self):
        return {
            'N': self.N,
            'sig_p': self.sig_p,
            'sig_b': self.sig_b,
            'sig_m': self.sig_m
        }
    
    def to_dict(self):
        return {
            'n': self.n,
            'F': self.F,
            'dt': self.dt,
            'dt_m': self.dt_m,
            'sig_m': self.sig_m,
            's': self.s,
            'obs_strategy': self.obs_strategy,
            'obs_indices': self.obs_indices,
            'N': self.N,
            'sig_p': self.sig_p,
            'sig_b': self.sig_b,
            'tm': self.tm,
            'k': self.k,
            'seed': self.seed,
            'spinup_time': self.spinup_time,
            'n_samples': self.n_samples,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'patience': self.patience,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'max_len': self.max_len,
            'nt': self.nt,
            'nt_m': self.nt_m,
            'ind_m': self.ind_m,
            'use_standard_eakf': self.use_standard_eakf,
            'eakf_eigenval_threshold': self.eakf_eigenval_threshold,
            'eakf_fallback_alpha': self.eakf_fallback_alpha
        }
    
    def set_random_seeds(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def get_model_path(self) -> str:
        return f"transformer_residual_enkf_tm{self.tm}_k{self.k}_seed{self.seed}.pth"

    def get_result_paths(self, k: int) -> dict:
        base_name = f"three_methods_comparison_tm{self.tm}_k{k}_seed{self.seed}"
        return {
            'npz': f"results/{base_name}.npz",
            'csv': f"results/{base_name}.csv",
            'png': f"results/{base_name}.png"
        }

    def print_summary(self):
        print("=" * 60)
        print("Experiment Configuration Summary")
        print("=" * 60)
        print(f"Dynamics: n={self.n}, F={self.F}, dt={self.dt}")
        print(f"Observation: dt_m={self.dt_m}, sig_m={self.sig_m}")
        print(f"EnKF: N={self.N}, sig_p={self.sig_p}, sig_b={self.sig_b}")
        print(f"Experiment: tm={self.tm}s, k={self.k}, seed={self.seed}")
        print(f"Training: n_samples={self.n_samples}, batch_size={self.batch_size}, epochs={self.epochs}")
        print(f"Model: d_model={self.d_model}, num_heads={self.num_heads}, num_layers={self.num_layers}")
        print(f"Derived: nt={self.nt}, nt_m={self.nt_m}")
        print("=" * 60)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()


 


def get_high_resolution_config() -> ExperimentConfig:
    return ExperimentConfig(
        tm=10.0,
        n_samples=200,
        epochs=200,
        batch_size=128
    )


def get_low_noise_config() -> ExperimentConfig:
    return ExperimentConfig(
        sig_p=0.02,
        sig_m=0.05,
        sig_b=0.5
    )


def get_default_inference_config() -> ExperimentConfig:
    return ExperimentConfig(
        tm=5.0,
        sig_p=0.5,
        sig_b=1.0,
        sig_m=0.15,
        dt_m=0.2,
        N=30,
        seed=30,
        s=9
    )


def get_training_config(tm=5, n_samples=100, epochs=100, batch_size=64, **kwargs) -> ExperimentConfig:
    config = ExperimentConfig(
        tm=tm,
        n_samples=n_samples,
        epochs=epochs,
        batch_size=batch_size,
        sig_p=0.02,
        sig_b=1.0,
        sig_m=0.1,
        dt_m=0.2,
        N=30,
        seed=30
    )
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def get_inflation_factor_configs() -> List[ExperimentConfig]:
    inflation_factors = [0.95, 1.00, 1.02, 1.05, 1.08]
    configs = []
    
    for i, inflation_factor in enumerate(inflation_factors):
        config = get_default_inference_config()
        config.inflation_factor = inflation_factor
        config.seed = 20
        configs.append(config)
    
    return configs


def get_inflation_scan_config(inflation_factor: float, **kwargs) -> ExperimentConfig:
    config = get_default_inference_config()
    config.inflation_factor = inflation_factor
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config



