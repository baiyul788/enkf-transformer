import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import time
from config.experiment_config import get_default_inference_config
from main import run_four_methods_comparison


EXPERIMENTS = [
    {'type': 's_sensitivity', 'param': 's', 'values': [9, 12, 18]},
    {'type': 'N_sensitivity', 'param': 'N', 'values': [30, 50, 70]},
    {'type': 'sig_p_sensitivity', 'param': 'sig_p', 'values': [0.45, 0.5, 0.6, 0.65]},
    {'type': 'sig_m_sensitivity', 'param': 'sig_m', 'values': [0.05, 0.15, 0.3, 0.35, 0.4]}
]


def run_experiment(param_name, param_value):
    config = get_default_inference_config()
    setattr(config, param_name, param_value)
    config.__post_init__()
    
    print(f"Running {param_name}={param_value}...", end=" ")
    
    try:
        results = run_four_methods_comparison(config, k=5, save_results=True)
        print(f"✓ EnKF={results['rmse_enkf']:.3f} EAKF-TR={results['rmse_eakf_tr']:.3f}")
        
        return {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'N': config.N, 's': config.s, 'sig_p': config.sig_p, 'sig_m': config.sig_m,
            'rmse_enkf': results['rmse_enkf'],
            'rmse_eakf': results['rmse_eakf'],
            'rmse_enkf_tr': results['rmse_enkf_tr'],
            'rmse_eakf_tr': results['rmse_eakf_tr'],
            'eakf_vs_enkf': results['eakf_vs_enkf'],
            'improve_enkf_tr': results['improve_enkf_tr'],
            'eakf_tr_vs_enkf': results['eakf_tr_vs_enkf']
        }
    except Exception as e:
        print(f"✗ {e}")
        return {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'N': config.N, 's': config.s, 'sig_p': config.sig_p, 'sig_m': config.sig_m,
            'rmse_enkf': None, 'rmse_eakf': None, 'rmse_enkf_tr': None, 'rmse_eakf_tr': None,
            'eakf_vs_enkf': None, 'improve_enkf_tr': None, 'eakf_tr_vs_enkf': None,
            'error': str(e)
        }


def main():
    print("="*80)
    print("Parameter Sensitivity Scan")
    print("="*80)
    
    all_results = []
    start_time = time.time()
    
    for exp_group in EXPERIMENTS:
        exp_type = exp_group['type']
        param_name = exp_group['param']
        param_values = exp_group['values']
        
        print(f"\n{exp_type}: {param_name} ∈ {param_values}")
        
        for param_value in param_values:
            result = run_experiment(param_name, param_value)
            result['experiment_type'] = exp_type
            all_results.append(result)
    
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(all_results)
    
    df.to_csv('results/baseline_parameter_sensitivity_scan.csv', index=False, float_format='%.6f')
    
    for exp_type in df['experiment_type'].unique():
        subset = df[df['experiment_type'] == exp_type]
        subset.to_csv(f'results/{exp_type}_results.csv', index=False, float_format='%.6f')
    
    elapsed = time.time() - start_time
    success = len([r for r in all_results if r['rmse_enkf'] is not None])
    
    print(f"\n{'='*80}")
    print(f"Completed: {success}/{len(all_results)} experiments ({elapsed/60:.1f} min)")
    print(f"Results saved to results/")
    print(f"{'='*80}")
    
    return df


if __name__ == "__main__":
    main()
