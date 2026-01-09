
import torch
import numpy as np
from typing import Dict, Tuple
from config.experiment_config import ExperimentConfig
from models.trainer import TransformerTrainer
from transformer_lorenz96 import TransformerModel
import time


def create_model(config: ExperimentConfig) -> TransformerModel:
    model_config = config.get_model_config()
    model = TransformerModel(**model_config)
    print(f"Model created: {model_config}")
    return model


def train_transformer_with_config(
    config: ExperimentConfig,
    train_input: np.ndarray,
    train_target: np.ndarray,
    val_input: np.ndarray,
    val_target: np.ndarray,
    model_save_path: str = None
) -> Tuple[TransformerModel, Dict]:
    print("=" * 60)
    print("Training Transformer model with new trainer")
    print("=" * 60)
    
    model = create_model(config)
    
    trainer = TransformerTrainer(config, model)
    
    train_loader, val_loader = trainer.prepare_data(
        train_input, train_target, val_input, val_target
    )
    
    history = trainer.train(train_loader, val_loader)
    
    if model_save_path:
        trainer.save_model(model_save_path)
    
    summary = trainer.get_training_summary()
    print("\nTraining summary:")
    print(f"  Best validation loss: {summary['best_val_loss']:.6f}")
    print(f"  Best validation RMSE: {summary['best_val_rmse']:.6f}")
    print(f"  Total epochs: {summary['total_epochs']}")
    print(f"  Model parameters: {summary['model_parameters']:,}")
    
    return model, history


def load_and_predict(
    config: ExperimentConfig,
    model_path: str,
    input_data: np.ndarray
) -> np.ndarray:
    model = create_model(config)
    
    trainer = TransformerTrainer(config, model)
    
    trainer.load_model(model_path)
    
    predicted_residual = trainer.predict(input_data)
    
    return predicted_residual


def compare_training_methods(
    config: ExperimentConfig,
    train_input: np.ndarray,
    train_target: np.ndarray,
    val_input: np.ndarray,
    val_target: np.ndarray
) -> Dict:
    print("=" * 60)
    print("Comparing new and old training methods")
    print("=" * 60)
    
    print("\n1. New trainer method:")
    start_time = time.time()
    model_new, history_new = train_transformer_with_config(
        config, train_input, train_target, val_input, val_target
    )
    time_new = time.time() - start_time
    
    print("\n2. Old training method:")
    from transformer_lorenz96 import train_transformer_full_state
    
    start_time = time.time()
    model_old, history_old = train_transformer_full_state(
        train_input, train_target, val_input, val_target,
        **config.get_training_config()
    )
    time_old = time.time() - start_time
    
    comparison = {
        'new_method': {
            'time': time_new,
            'best_val_loss': min(history_new['val_loss']),
            'best_val_rmse': min(history_new['val_rmse']),
            'total_epochs': len(history_new['train_loss'])
        },
        'old_method': {
            'time': time_old,
            'best_val_loss': min(history_old['val_loss']),
            'best_val_rmse': min(history_old['val_rmse']),
            'total_epochs': len(history_old['train_loss'])
        }
    }
    
    print("\nComparison results:")
    print(f"  New method time: {time_new:.1f}s")
    print(f"  Old method time: {time_old:.1f}s")
    print(f"  Time difference: {((time_new - time_old) / time_old * 100):+.1f}%")
    print(f"  New method best validation RMSE: {comparison['new_method']['best_val_rmse']:.6f}")
    print(f"  Old method best validation RMSE: {comparison['old_method']['best_val_rmse']:.6f}")
    
    return comparison


def train_enkf_transformer_with_config(
    config: ExperimentConfig,
    model_save_path: str = None
) -> Tuple[TransformerModel, Dict]:
    print("=" * 80)
    print("EnKF Transformer Training")
    print("=" * 80)

    from data_generator import generate_enkf_residual_training_data
    train_input, train_target, val_input, val_target = generate_enkf_residual_training_data(
        n_samples=config.n_samples,
        tm=config.tm,
        dt_m=config.dt_m,
        k=config.k,
        N=config.N,
        sig_p=config.sig_p,
        sig_b=config.sig_b,
        sig_m=config.sig_m,
        spinup_time=config.spinup_time,
        seed=config.seed,
        verbose=True,
        progress_every=10,
        enable_cache=True,
        cache_dir="results"
    )

    if model_save_path is None:
        model_save_path = (
            f"models/transformer_residual_enkf_tm{config.tm}_k{config.k}_"
            f"sp{config.sig_p}_seed{config.seed}.pth"
        )

    model, history = train_transformer_with_config(
        config, train_input, train_target, val_input, val_target, model_save_path
    )

    if model_save_path:
        print(f"EnKF model saved to: {model_save_path}")

    return model, history


def train_eakf_transformer_with_config(
    config: ExperimentConfig,
    model_save_path: str = None
) -> Tuple[TransformerModel, Dict]:
    print("=" * 80)
    print("EAKF Transformer Training")
    print("=" * 80)

    from data_generator import generate_eakf_residual_training_data
    train_input, train_target, val_input, val_target = generate_eakf_residual_training_data(
        n_samples=config.n_samples,
        tm=config.tm,
        dt_m=config.dt_m,
        k=config.k,
        N=config.N,
        sig_p=config.sig_p,
        sig_b=config.sig_b,
        sig_m=config.sig_m,
        spinup_time=config.spinup_time,
        seed=config.seed,
        verbose=True,
        progress_every=10,
        enable_cache=True,
        cache_dir="results"
    )

    if model_save_path is None:
        model_save_path = (
            f"models/transformer_residual_eakf_tm{config.tm}_k{config.k}_"
            f"sp{config.sig_p}_seed{config.seed}.pth"
        )

    model, history = train_transformer_with_config(
        config, train_input, train_target, val_input, val_target, model_save_path
    )

    if model_save_path:
        print(f"EAKF model saved to: {model_save_path}")

    return model, history
