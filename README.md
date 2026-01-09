# Transformer-Predicted Residuals for Data Assimilation

[![Reproducibility](https://img.shields.io/badge/Reproducibility-±10%25%20variation%20expected-blue)](FAQ.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

> **📌 Note**: Results may vary by ±10% across platforms while maintaining consistent trends. See [FAQ](FAQ.md#q1-why-are-my-numerical-results-different-from-the-paper) for details.

Code for paper: *Transformer-Predicted Residuals for Weakly Coupled Integration Paradigm in Ensemble Data Assimilation*

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib pandas scipy

# Run experiments
cd tenkf
python test_all.py              # Comprehensive test (~5 min)
python main.py                  # Generate data
python generate_all_figures.py  # Generate figures
```

## 📊 Expected Results

**Baseline** (N=30, s=9, σ_p=0.5, σ_m=0.15, seed=30):

| Method | Paper RMSE | Typical Range | Improvement |
|--------|------------|---------------|-------------|
| EnKF | 3.681 | 3.4-3.8 | Baseline |
| EAKF | 2.739 | 2.6-2.8 | ~23-26% |
| EnKF-TR | 2.302 | 2.1-2.4 | ~37-40% |
| EAKF-TR | 1.620 | 1.5-1.7 | ~54-58% |

**Key**: Relative ranking (EAKF-TR > EnKF-TR > EAKF > EnKF) should be preserved.

## 📁 Project Structure

```
tenkf/
├── main.py                          # Main experiment script
├── test_all.py                      # Comprehensive test suite
├── lorenz96_model.py                # Lorenz-96 model
├── enkf_module.py / eakf_module.py  # Data assimilation algorithms
├── transformer_lorenz96.py          # Transformer model
├── generate_all_figures.py          # Figure generation
├── config/experiment_config.py      # Configuration
└── models/                          # Pre-trained models
```

## 📖 Documentation

- **[FAQ.md](FAQ.md)** - Common questions and troubleshooting
- **Full README** - See below for detailed documentation

## 🔬 Full Experiments

```bash
cd tenkf

# 1. Train models (optional, pre-trained models provided)
python run_batch_tr_training.py

# 2. Run all experiments
python run_four_methods_batch.py

# 3. Parameter sensitivity analysis
python run_baseline_parameter_sensitivity_scan.py

# 4. Generate all figures
python generate_all_figures.py --heatmaps
```

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/enkf-transformer/issues)
- **Email**: fanmanhong@nwnu.edu.cn

## 📄 Citation

```bibtex
@article{fan2024transformer,
  title={Transformer-Predicted Residuals for Weakly Coupled Integration 
         Paradigm in Ensemble Data Assimilation},
  author={Fan, Manhong and Bai, Yonglong and Ding, Lin and 
          Xiao, Qian and Yu, Qinghe},
  journal={Science China Earth Sciences},
  year={2024}
}
```

---

## Detailed Documentation

### Main Features

✅ Complete data assimilation workflow (EnKF/EAKF)  
✅ Transformer residual prediction model  
✅ Parameter sensitivity analysis framework  
✅ Paper-quality figure generation  
✅ Pre-trained models included  

### Project Structure (Detailed)

```
enkf-transformer/
├── README.md                                    # This file
├── FAQ.md                                       # Common questions
├── tenkf/                                       # Core code
│   ├── main.py                                  # Main entry point
│   ├── test_all.py                              # Comprehensive test suite
│   ├── lorenz96_model.py                        # Lorenz-96 model
│   ├── enkf_module.py                           # EnKF algorithm
│   ├── eakf_module.py                           # EAKF algorithm
│   ├── transformer_lorenz96.py                  # Transformer model
│   ├── transformer_utils.py                     # Transformer utilities
│   ├── data_generator.py                        # Training data generator
│   ├── visualization.py                         # Visualization module
│   ├── train_transformer.py                     # Transformer training
│   ├── run_baseline_parameter_sensitivity_scan.py  # Sensitivity analysis
│   ├── run_batch_tr_training.py                 # Batch training
│   ├── generate_all_figures.py                  # Figure generation
│   ├── config/
│   │   └── experiment_config.py                 # Configuration
│   ├── models/
│   │   ├── trainer.py                           # Trainer class
│   │   ├── training_utils.py                    # Training utilities
│   │   └── *.pth                                # Pre-trained models
│   └── results/                                 # Experiment results
└── .github/
    └── ISSUE_TEMPLATE/
        └── reproducibility_question.md          # Issue template
```

### Configuration Parameters

**Baseline** (in `config/experiment_config.py`):
- `N = 30` - Ensemble size
- `s = 9` - Number of observations (out of 36 states)
- `tm = 5.0` - Simulation time
- `k = 5` - Downsampling rate
- `σ_p = 0.5` - Model error std
- `σ_m = 0.15` - Observation error std
- `seed = 30` - Random seed

**Sensitivity ranges**:
- N: [30, 50, 70]
- s: [9, 12, 18]
- σ_p: [0.45, 0.5, 0.6, 0.65]
- σ_m: [0.05, 0.15, 0.3, 0.35, 0.4]

### Figure Generation

```bash
cd tenkf

# Generate all figures
python generate_all_figures.py

# Generate specific figures
python generate_all_figures.py --figures 4 5 6

# Include heatmaps
python generate_all_figures.py --heatmaps

# Specific parameter heatmaps
python generate_all_figures.py --heatmaps --heatmap-params s N
```

**Figures**:
- **Figure 4**: Methods comparison (Hovmöller diagrams)
- **Figure 5**: Error time series and uncertainty analysis
- **Figure 6**: Multi-variable state reconstruction
- **Figure 7**: Parameter sensitivity analysis

### Reproducibility Notes

**Why do results vary?**
1. Random number generator differences across platforms
2. Floating-point arithmetic precision
3. Chaotic system sensitivity (Lorenz-96)
4. PyTorch/NumPy version differences

**What should be consistent?**
- ✅ Relative ranking: EAKF-TR > EnKF-TR > EAKF > EnKF
- ✅ Improvement trends: ~23%, ~38%, ~55%
- ✅ Sensitivity curve patterns
- ✅ Spatial error distributions

**Validation**:
```bash
cd tenkf
python test_all.py

# Check:
# 1. EAKF is 20-30% better than EnKF
# 2. Transformer methods are 30-50% better than baselines
# 3. EAKF-TR achieves best performance
```

See [FAQ](FAQ.md) for more details.

### License

MIT License - see [LICENSE](LICENSE) file

### Acknowledgements

This work was supported by:
- National Natural Science Foundation of China (Grant No. 42461053)
- Department of Education of Gansu Province (Grant Nos. 2023B-064 and 2024QB-014)
- Natural Science Foundation of Gansu Province (Grant No. 25JRRA012)
