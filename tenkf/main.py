import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import time
import torch

from enkf_module import run_complete_enkf_experiment
from transformer_lorenz96 import TransformerModel
from config.experiment_config import get_training_config, get_default_inference_config
from transformer_utils import apply_transformer_correction


def run_four_methods_comparison(config, k: int = 5, save_results: bool = True) -> dict:
    from eakf_module import run_complete_eakf_experiment

    np.random.seed(config.seed)
    xTrue, xa_enkf, _, yo = run_complete_enkf_experiment(config)
    np.random.seed(config.seed)
    _, xa_eakf, _, _ = run_complete_eakf_experiment(config)
    enkf_model_path = (
        f"models/transformer_residual_enkf_"
        f"N{config.N}_s{config.s}_tm{config.tm:.1f}_k{config.k}_"
        f"sp{config.sig_p}_sm{config.sig_m}_seed{config.seed}.pth"
    )
    xa_enkf_tr = apply_transformer_correction(xa_enkf, enkf_model_path, k=k)

    eakf_model_path = (
        f"models/transformer_residual_eakf_"
        f"N{config.N}_s{config.s}_tm{config.tm:.1f}_k{config.k}_"
        f"sp{config.sig_p}_sm{config.sig_m}_seed{config.seed}.pth"
    )
    xa_eakf_tr = apply_transformer_correction(xa_eakf, eakf_model_path, k=k)
    has_eakf_tr = True

    xTrue_ds = xTrue[:, ::k]
    rmse_enkf = np.sqrt(np.mean((xa_enkf[:, ::k] - xTrue_ds) ** 2, axis=0)).mean()
    rmse_eakf = np.sqrt(np.mean((xa_eakf[:, ::k] - xTrue_ds) ** 2, axis=0)).mean()
    rmse_enkf_tr = np.sqrt(np.mean((xa_enkf_tr - xTrue_ds) ** 2, axis=0)).mean()
    rmse_eakf_tr = (
        np.sqrt(np.mean((xa_eakf_tr - xTrue_ds) ** 2, axis=0)).mean()
        if has_eakf_tr else None
    )

    improve_enkf_tr = (rmse_enkf - rmse_enkf_tr) / rmse_enkf * 100
    eakf_vs_enkf = (rmse_enkf - rmse_eakf) / rmse_enkf * 100
    eakf_tr_vs_enkf = (
        (rmse_enkf - rmse_eakf_tr) / rmse_enkf * 100 if has_eakf_tr else None
    )

    if save_results:
        os.makedirs('results', exist_ok=True)
        tag = (
            f"N{config.N}_s{config.s}_tm{config.tm:.1f}_k{k}_"
            f"sp{config.sig_p}_sm{config.sig_m}_seed{config.seed}"
        )

        time_series_data = {
            'rmse_enkf_t': np.sqrt(np.mean((xa_enkf[:, ::k] - xTrue_ds) ** 2, axis=0)),
            'rmse_eakf_t': np.sqrt(np.mean((xa_eakf[:, ::k] - xTrue_ds) ** 2, axis=0)),
            'rmse_enkf_tr_t': np.sqrt(np.mean((xa_enkf_tr - xTrue_ds) ** 2, axis=0)),
            'xTrue': xTrue,
            'xa_enkf': xa_enkf,
            'xa_eakf': xa_eakf,
            'xa_enkf_tr': xa_enkf_tr,
            'yo': yo,
        }
        if has_eakf_tr:
            time_series_data['rmse_eakf_tr_t'] = np.sqrt(np.mean((xa_eakf_tr - xTrue_ds) ** 2, axis=0))
            time_series_data['xa_eakf_tr'] = xa_eakf_tr
        npz_path = f'results/four_methods_{tag}.npz'
        np.savez(npz_path, **time_series_data)

        t = np.arange(time_series_data['rmse_enkf_t'].shape[0]) * (config.dt * config.k)
        if has_eakf_tr:
            csv_array = np.column_stack([
                t,
                time_series_data['rmse_enkf_t'],
                time_series_data['rmse_eakf_t'],
                time_series_data['rmse_enkf_tr_t'],
                time_series_data['rmse_eakf_tr_t'],
            ])
            header = "t,rmse_enkf,rmse_eakf,rmse_enkf_tr,rmse_eakf_tr"
        else:
            csv_array = np.column_stack([
                t,
                time_series_data['rmse_enkf_t'],
                time_series_data['rmse_eakf_t'],
                time_series_data['rmse_enkf_tr_t'],
            ])
            header = "t,rmse_enkf,rmse_eakf,rmse_enkf_tr"
        csv_path = f'results/four_methods_{tag}.csv'
        np.savetxt(csv_path, csv_array, delimiter=",", header=header, comments="")

        summary_data = {
            'method': ['EnKF', 'EAKF', 'EnKF+Transformer'],
            'rmse': [rmse_enkf, rmse_eakf, rmse_enkf_tr],
            'improvement_vs_enkf': [0, eakf_vs_enkf, improve_enkf_tr]
        }
        if has_eakf_tr:
            summary_data['method'].append('EAKF+Transformer')
            summary_data['rmse'].append(rmse_eakf_tr)
            summary_data['improvement_vs_enkf'].append(eakf_tr_vs_enkf)
        import pandas as pd, json
        summary_path = f'results/summary_{tag}.csv'
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)

        meta = {
            'params': {
                'N': config.N,
                's': config.s,
                'tm': config.tm,
                'k': k,
                'sig_p': config.sig_p,
                'sig_m': config.sig_m,
                'seed': config.seed,
                'inflation_factor': getattr(config, 'inflation_factor', None),
            },
            'metrics': {
                'rmse_enkf': rmse_enkf,
                'rmse_eakf': rmse_eakf,
                'rmse_enkf_tr': rmse_enkf_tr,
                'rmse_eakf_tr': rmse_eakf_tr,
                'improve_enkf_tr_%': improve_enkf_tr,
                'eakf_vs_enkf_%': eakf_vs_enkf,
                'eakf_tr_vs_enkf_%': eakf_tr_vs_enkf,
            },
            'files': {
                'npz': npz_path,
                'csv': csv_path,
                'summary_csv': summary_path,
            }
        }
        json_path = f'results/summary_{tag}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        


    return {
        'rmse_enkf': rmse_enkf,
        'rmse_eakf': rmse_eakf,
        'rmse_enkf_tr': rmse_enkf_tr,
        'rmse_eakf_tr': rmse_eakf_tr,
        'improve_enkf_tr': improve_enkf_tr,
        'eakf_vs_enkf': eakf_vs_enkf,
        'eakf_tr_vs_enkf': eakf_tr_vs_enkf,
        'xTrue': xTrue,
        'xa_enkf': xa_enkf,
        'xa_eakf': xa_eakf,
        'xa_enkf_tr': xa_enkf_tr,
        'xa_eakf_tr': xa_eakf_tr if has_eakf_tr else None,
        'yo': yo,
    }
def generate_analysis_report(results, config):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Experiment Results")
    report_lines.append("=" * 80)
    report_lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Config: tm={config.tm}, seed={config.seed}")
    report_lines.append("")
    
    report_lines.append("1. Performance Comparison:")
    report_lines.append("-" * 40)
    report_lines.append(f"EnKF RMSE:              {results['rmse_enkf']:.6f}")
    report_lines.append(f"EAKF RMSE:              {results['rmse_eakf']:.6f}")
    report_lines.append(f"EnKF+Transformer RMSE:  {results['rmse_enkf_tr']:.6f}")
    if results['rmse_eakf_tr'] is not None:
        report_lines.append(f"EAKF+Transformer RMSE:  {results['rmse_eakf_tr']:.6f}")
    report_lines.append("")
    
    report_lines.append("2. Improvement vs EnKF Baseline:")
    report_lines.append("-" * 40)
    report_lines.append(f"EnKF -> EnKF+Transformer: {results['improve_enkf_tr']:+.2f}%")
    report_lines.append(f"EAKF vs EnKF:           {results['eakf_vs_enkf']:+.2f}%")
    if results['eakf_tr_vs_enkf'] is not None:
        report_lines.append(f"EAKF+Transformer vs EnKF: {results['eakf_tr_vs_enkf']:+.2f}%")
    report_lines.append("")
    

    
    report_path = f'results/analysis_report_tm{config.tm}_k5_seed{config.seed}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print("1. Performance Comparison:")
    print("-" * 40)
    print(f"EnKF RMSE:              {results['rmse_enkf']:.6f}")
    print(f"EAKF RMSE:              {results['rmse_eakf']:.6f}")
    print(f"EnKF+Transformer RMSE:  {results['rmse_enkf_tr']:.6f}")
    if results['rmse_eakf_tr'] is not None:
        print(f"EAKF+Transformer RMSE:  {results['rmse_eakf_tr']:.6f}")
    print("")

    print("2. Improvement vs EnKF Baseline:")
    print("-" * 40)
    print(f"EnKF -> EnKF+Transformer: {results['improve_enkf_tr']:+.2f}%")
    print(f"EAKF vs EnKF:           {results['eakf_vs_enkf']:+.2f}%")
    if results['eakf_tr_vs_enkf'] is not None:
        print(f"EAKF+Transformer vs EnKF: {results['eakf_tr_vs_enkf']:+.2f}%")
    


if __name__ == "__main__":
    config = get_default_inference_config()
    results = run_four_methods_comparison(config, k=5, save_results=True)

    generate_analysis_report(results, config)
    
    try:
        from visualization import plot_individual_subplots, plot_methods_comparison_grid
        
        saved_paths = plot_individual_subplots(config, results, 'results')
        for path in saved_paths:
            print(f"   - {path}")
        
        comparison_path = plot_methods_comparison_grid(config, results, 'results')
        print(f"Comparison grid: {comparison_path}")
        
    except ImportError as e:
        print(f"Visualization import failed: {e}")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
