# Frequently Asked Questions (FAQ)

## Reproducibility & Results

### Q1: Why are my numerical results different from the paper?

**A**: This is **normal and expected**! Results can vary by 5-10% due to random number generators, floating-point precision, and chaotic system sensitivity.

**What matters**: Relative ranking (EAKF-TR > EnKF-TR > EAKF > EnKF) and improvement trends (~23%, ~38%, ~55%) should be consistent.

**Quick check**:
```bash
cd tenkf
python test_all.py
```

---

### Q2: Are my results acceptable?

**A**: Yes, if:
- ✅ Relative ranking is preserved
- ✅ Improvement percentages are within ±10% of paper values
- ✅ No error messages

**Expected ranges** (baseline: N=30, s=9, σ_p=0.5, σ_m=0.15):

| Method | Paper | Acceptable Range |
|--------|-------|------------------|
| EnKF | 3.681 | 3.4-3.8 |
| EAKF | 2.739 | 2.6-2.8 |
| EnKF-TR | 2.302 | 2.1-2.4 |
| EAKF-TR | 1.620 | 1.5-1.7 |

---

## Installation & Setup

### Q3: What Python version should I use?

**A**: Python 3.8 or 3.9 recommended. Python 3.10+ should work but may have minor compatibility issues.

---

### Q4: Do I need a GPU?

**A**: No. All experiments run on CPU. Pre-trained models are provided.

---

### Q5: Installation fails with "No module named 'torch'"

**A**: Install PyTorch:
```bash
pip install torch numpy matplotlib pandas scipy
```

---

## Running Experiments

### Q6: How long does it take?

**A**: 
- Quick test: ~2-5 minutes
- Single experiment: ~5-10 minutes
- Full sensitivity scan: ~2-4 hours

---

### Q7: Where are results saved?

**A**: 
- `tenkf/results/*.npz` - Experiment data
- `tenkf/results/*.csv` - Sensitivity results
- `tenkf/results/*.png` - Figures

---

### Q8: How do I generate figures?

**A**:
```bash
cd tenkf
python main.py                    # Generate data first
python generate_all_figures.py   # Then generate figures
```

---

## Troubleshooting

### Q9: Error: "Data file not found"

**A**: Run experiments first:
```bash
cd tenkf
python main.py
```

---

### Q10: Results are all NaN or Inf

**A**: Check parameters:
- Ensemble size N ≥ 20
- Observation error σ_m ≤ 0.5
- Model error 0.3 ≤ σ_p ≤ 0.8

---

### Q11: Code runs but RMSE > 10

**A**: Verify:
1. Pre-trained models exist in `tenkf/models/`
2. Model filenames match configuration
3. Random seed is set correctly

---

## Advanced

### Q12: Can I use this for other dynamical systems?

**A**: Yes! Modify:
1. Model implementation (like `lorenz96_model.py`)
2. Observation operator
3. Transformer input dimension
4. Retrain models

---

### Q13: How to speed up experiments?

**A**:
- Use smaller ensemble (N=20)
- Reduce simulation time (tm=3.0)
- Run parameter sweeps in parallel
- Use pre-trained models

---

## Citation

### Q14: How should I cite this work?

**A**:
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

## Still Have Questions?

1. Check README.md for detailed documentation
2. Search [existing issues](https://github.com/your-username/enkf-transformer/issues)
3. Open a [new issue](https://github.com/your-username/enkf-transformer/issues/new)
4. Email: fanmanhong@nwnu.edu.cn

---

**Last Updated**: January 2026
