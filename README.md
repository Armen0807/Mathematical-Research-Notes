# Exhaustion Framework: Quantitative Convergence & Risk Metrics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Research Status](https://img.shields.io/badge/Research-Active-red)]()

## 🚀 Overview

This repository hosts the reference implementation and research papers for the **Exhaustion Framework**, a novel mathematical approach to quantifying asymptotic convergence in heavy-tailed environments.

Standard uniform metrics (like the classic Kolmogorov distance) often fail to capture convergence rates in financial data characterized by infinite variance or skewness. This framework introduces **Weighted Kolmogorov Metrics** ($d_{K,h,q}$) via the concept of *exhaustion functions*.

**Key Results:**
* **Restored Convergence:** We prove and demonstrate **$O(n^{-1/2})$ convergence rates** even for heavy-tailed distributions (Pareto, Student-t) where the third moment diverges.
* **Central Focus:** The metric naturally downweights tail noise to focus on the central approximation accuracy, critical for robust VaR (Value at Risk) backtesting.
* **Production-Grade Implementation:** Optimized Python/SciPy code to compute these metrics efficiently on large datasets.

---

## 📄 Research Papers

The theoretical foundations and proofs are detailed in the following papers included in the `papers/` directory:

* **[Petrosyan_2025_Weighted_Kolmogorov.pdf](papers/Petrosyan_2025_Weighted_Kolmogorov.pdf)**
    * *Topic:* Restoring Gaussian convergence rates in heavy-tailed risk models using weighted metrics.
    * *Key Finding:* Validates $O(n^{-1/2})$ rate for Student-t ($\nu=2.5$) and Pareto ($\alpha=2.8$).

* **[Petrosyan_2025_Exhaustion_Framework.pdf](papers/Petrosyan_2025_Exhaustion_Framework.pdf)**
    * *Topic:* A unified topological framework for defining convergence at infinity via exhaustion functions.
    * *Scope:* Extends to anisotropic spaces and spectral theory.

---

## 🛠 Installation & Usage

### Requirements
"""```bash
""pip install -r requirements.txt

import numpy as np
from exhaustion_framework.metric import weighted_kolmogorov_error

# 1. Generate Heavy-Tailed Data (e.g., Pareto distributed returns)
data = np.random.pareto(a=2.8, size=1000)

# 2. Define Evaluation Grid (Central region focus)
t_grid = np.linspace(-8, 8, 4001)

# 3. Compute Weighted Metric
# q=1.2 is chosen to satisfy the condition q > (2+delta)/2
weighted_err, uniform_err = weighted_kolmogorov_error(data, q=1.2, t_grid=t_grid)

print(f"Weighted Error: {weighted_err:.5f} (Captures central behavior)")
print(f"Uniform Error:  {uniform_err:.5f} (Dominated by tail noise)")

## 📊 Reproducibility & Experiments
To reproduce the convergence graphs (Figures 1 & 2 in the paper), run the simulation script provided. This validates the theoretical bounds derived in Petrosyan (2025).

python simulation.py --outdir experiments/outputs

Outputs: Results are saved in experiments/outputs/. The simulations compare Uniform vs. Weighted error decay for Student-t and Pareto distributions, confirming the theoretical slope of −1/2 for the weighted metric.

## 📚 Citation

If you use this framework or code in your research, please cite:
@article{petrosyan2025exhaustion,
  title={Measuring Asymptotic Convergence: A Unified Framework & Weighted Kolmogorov Metrics},
  author={Petrosyan, Armen},
  year={2025},
  publisher={GitHub},
  journal={SSRN Preprint}
}

## 👤 Author
Armen Petrosyan

Research Interests: Quantitative Finance, Stochastic Analysis, Heavy-Tailed Risk, Martingale Theory.

Background: Sorbonne University (Applied Mathematics) & EM Lyon (MSc in Management).

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE.md file for details.