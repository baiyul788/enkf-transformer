#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from config.experiment_config import get_training_config
from models.training_utils import train_enkf_transformer_with_config, train_eakf_transformer_with_config


PARAMETER_COMBINATIONS = [
    {'N': 30, 's': 9, 'sig_p': 0.5, 'sig_m': 0.15},
    {'N': 50, 's': 9, 'sig_p': 0.5, 'sig_m': 0.15},
    {'N': 70, 's': 9, 'sig_p': 0.5, 'sig_m': 0.15},
    {'N': 30, 's': 12, 'sig_p': 0.5, 'sig_m': 0.15},
    {'N': 30, 's': 18, 'sig_p': 0.5, 'sig_m': 0.15},
    {'N': 30, 's': 9, 'sig_p': 0.45, 'sig_m': 0.15},
    {'N': 30, 's': 9, 'sig_p': 0.6, 'sig_m': 0.15},
    {'N': 30, 's': 9, 'sig_p': 0.65, 'sig_m': 0.15},
    {'N': 30, 's': 9, 'sig_p': 0.5, 'sig_m': 0.05},
    {'N': 30, 's': 9, 'sig_p': 0.5, 'sig_m': 0.3},
    {'N': 30, 's': 9, 'sig_p': 0.5, 'sig_m': 0.35},
    {'N': 30, 's': 9, 'sig_p': 0.5, 'sig_m': 0.4},
]


def get_model_path(method, N, s, sig_p, sig_m, tm=5.0, seed=30):
    return f"models/transformer_residual_{method}_N{N}_s{s}_tm{tm:.1f}_k5_sp{sig_p}_sm{sig_m}_seed{seed}.pth"


def train_model(method, combo, tm=5.0, epochs=200, n_samples=100, seed=30):
    N, s, sig_p, sig_m = combo['N'], combo['s'], combo['sig_p'], combo['sig_m']
    model_path = get_model_path(method, N, s, sig_p, sig_m, tm, seed)
    
    if os.path.exists(model_path):
        print(f"Skip {method.upper()} N={N} s={s} sp={sig_p} sm={sig_m} (exists)")
        return True
    
    print(f"Training {method.upper()} N={N} s={s} sp={sig_p} sm={sig_m}...")
    
    config = get_training_config(tm=tm, n_samples=n_samples, epochs=epochs, batch_size=64, seed=seed)
    config.N, config.s, config.sig_p, config.sig_m = N, s, sig_p, sig_m
    
    os.makedirs('models', exist_ok=True)
    
    try:
        if method == 'enkf':
            train_enkf_transformer_with_config(config, model_path)
        else:
            train_eakf_transformer_with_config(config, model_path)
        print(f"  ✓ Saved: {model_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['enkf', 'eakf', 'both'], default='both')
    parser.add_argument("--force", action='store_true', help="Retrain existing models")
    args = parser.parse_args()
    
    methods = ['enkf', 'eakf'] if args.method == 'both' else [args.method]
    total = len(PARAMETER_COMBINATIONS) * len(methods)
    
    print(f"{'='*80}")
    print(f"Batch Training: {len(PARAMETER_COMBINATIONS)} configs × {len(methods)} methods = {total} models")
    print(f"{'='*80}")
    
    if args.force:
        for combo in PARAMETER_COMBINATIONS:
            for method in methods:
                path = get_model_path(method, combo['N'], combo['s'], combo['sig_p'], combo['sig_m'])
                if os.path.exists(path):
                    os.remove(path)
        print("Removed existing models (--force)")
    
    start_time = time.time()
    success, failed = 0, 0
    
    for i, combo in enumerate(PARAMETER_COMBINATIONS, 1):
        for method in methods:
            print(f"\n[{i*len(methods)}/{total}]", end=" ")
            if train_model(method, combo):
                success += 1
            else:
                failed += 1
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed: {success} success, {failed} failed ({elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
