import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from visualization import (
    plot_methods_comparison_grid,
    plot_error_and_uncertainty_analysis,
    plot_multivar_state_comparison,
    plot_parameter_sensitivity_grid
)


def load_baseline_data(save_dir='results'):
    """Load baseline experiment data"""
    data_file = f'{save_dir}/four_methods_N30_s9_tm5.0_k5_sp0.5_sm0.15_seed30.npz'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run main.py first.")
        return None
    return np.load(data_file)


def generate_figure4(save_dir='results'):
    data = load_baseline_data(save_dir)
    if data is None:
        return None
    
    from config.experiment_config import get_default_inference_config
    config = get_default_inference_config()
    
    results_data = {k: data[k] for k in ['xTrue', 'xa_enkf', 'xa_eakf', 'xa_enkf_tr', 'xa_eakf_tr']}
    results_data['yo'] = data.get('yo', None)
    
    print("Generating Figure 4...")
    path = plot_methods_comparison_grid(config, results_data, save_dir)
    print(f"✓ {path}")
    return path


def generate_figure5(save_dir='results'):
    data = load_baseline_data(save_dir)
    if data is None:
        return None
    
    from config.experiment_config import get_default_inference_config
    config = get_default_inference_config()
    
    xb_keys = ['xb_enkf', 'xb_eakf', 'xb_enkf_tr', 'xb_eakf_tr']
    xb_data = [data[k] if k in data.files else None for k in xb_keys]
    
    print("Generating Figure 5...")
    path = plot_error_and_uncertainty_analysis(
        config, data['xTrue'], data['xa_enkf'], data['xa_eakf'], 
        data['xa_enkf_tr'], data['xa_eakf_tr'], *xb_data,
        variable_idx=19, save_dir=save_dir
    )
    print(f"✓ {path}")
    return path


def generate_figure6(save_dir='results'):
    data = load_baseline_data(save_dir)
    if data is None:
        return None
    
    from config.experiment_config import get_default_inference_config
    config = get_default_inference_config()
    
    results_data = {k: data[k] for k in ['xTrue', 'xa_enkf', 'xa_eakf', 'xa_enkf_tr', 'xa_eakf_tr']}
    
    print("Generating Figure 6...")
    path = plot_multivar_state_comparison(config, results_data, save_dir)
    print(f"✓ {path}")
    return path


def generate_figure7(save_dir='results'):
    files = {
        's_sensitivity': f'{save_dir}/s_sensitivity_results.csv',
        'N_sensitivity': f'{save_dir}/N_sensitivity_results.csv',
        'sig_p_sensitivity': f'{save_dir}/sig_p_sensitivity_results.csv',
        'sig_m_sensitivity': f'{save_dir}/sig_m_sensitivity_results.csv'
    }
    
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run sensitivity scan first.")
            return None
    
    sensitivity_data = {k: np.genfromtxt(p, delimiter=',', names=True, dtype=None, encoding='utf-8') 
                        for k, p in files.items()}
    
    print("Generating Figure 7...")
    path = plot_parameter_sensitivity_grid(sensitivity_data, save_dir)
    print(f"✓ {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--figures', nargs='+', choices=['4', '5', '6', '7', 'all'],
                        default=['all'], help='Figures to generate')
    parser.add_argument('--save-dir', default='results', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    figures = ['4', '5', '6', '7'] if 'all' in args.figures else args.figures
    
    print("="*80)
    print(f"Generating figures: {', '.join(figures)}")
    print("="*80)
    
    generators = {'4': generate_figure4, '5': generate_figure5, 
                  '6': generate_figure6, '7': generate_figure7}
    
    results = {f: generators[f](args.save_dir) for f in figures}
    success = sum(1 for v in results.values() if v is not None)
    
    print(f"\n{'='*80}")
    print(f"Completed: {success}/{len(results)} figures")
    print("="*80)
    
    return 0 if success == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
