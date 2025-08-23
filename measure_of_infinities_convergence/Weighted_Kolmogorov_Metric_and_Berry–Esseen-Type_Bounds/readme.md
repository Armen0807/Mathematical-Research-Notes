# Weighted Kolmogorov Metric and Berry-Esseen-Type Bounds

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Armen Petrosyan
**Date:** August 23, 2025

##  Abstract

This research introduces a novel **weighted Kolmogorov metric**, $d_{K,h,q}$, designed to measure the distance between probability distributions by focusing on their central behavior while down-weighting their tails. The weighting is controlled by an **exhaustion function** $h$ (e.g., $h(t)=|t|$).

The primary result is a Berry-Esseen-type theorem demonstrating that this new metric restores the classic Gaussian convergence rate of **$O(n^{-1/2})$** for normalized sums of i.i.d. random variables under the mild moment condition $\mathbb{E}|X|^{2+\delta} < \infty$. This is a significant improvement over the standard uniform metric, which yields a slower rate of $n^{-\delta/2}$ under the same conditions. Our approach provides a powerful tool for analyzing convergence for heavy-tailed distributions where the third moment may not exist.

---

##  The Core Idea

### The Problem
The classical Berry-Esseen theorem gives a convergence rate of $O(n^{-1/2})$ for the Central Limit Theorem but requires a finite third moment ($\mathbb{E}|X|^3 < \infty$). For many important distributions (like Student's t or Pareto with low degrees of freedom), this condition is not met, and the convergence rate in the standard uniform metric is much slower.

### Our Solution
We propose changing the metric itself. Instead of giving equal importance to all points, we introduce a weight that focuses the error measurement on the center of the distribution. Our **weighted Kolmogorov metric** is defined as:

$$
d_{K,h,q}(F,G) := \sup_{t\in\mathbb{R}} \frac{|F(t)-G(t)|}{(1+h(t))^q}
$$

By choosing the weight exponent $q$ appropriately ($q > (2+\delta)/2$), we can effectively ignore the slow convergence in the tails and recover the fast $O(n^{-1/2})$ rate in the region that matters most for many practical applications.

## How to Cite
If you find this work useful in your research, please consider citing the main document:

@misc{Petrosyan_WeightedKolmogorov_2025,
  author       = {Petrosyan, Armen},
  title        = {Weighted Kolmogorov Metric and Berry-Esseen-Type Bounds: $n^{-1/2}$ rates under $2+\delta$ moments via exhaustion functions},
  year         = {2025},
  howpublished = {\url{[https://github.com/Armen0807/Mathematical-Research-Notes/tree/main/measure_of_infinities_convergence/Weighted_Kolmogorov_Metric_and_Berry–Esseen-Type_Bounds](https://github.com/Armen0807/Mathematical-Research-Notes/tree/main/measure_of_infinities_convergence/Weighted_Kolmogorov_Metric_and_Berry–Esseen-Type_Bounds)}},

}

---

##  Numerical Validation

This repository includes a Python script to validate the theoretical findings. We simulate sums of random variables from heavy-tailed distributions (Student's t and Pareto) and compare the convergence rate in the standard uniform metric versus our new weighted metric.

The results confirm our theory: the weighted metric consistently exhibits a convergence rate close to $n^{-1/2}$, while the uniform metric's rate is significantly slower.

![Student t(ν=2.5): uniform vs weighted errors](out\compare_student.png)
*Figure: Comparison of convergence rates for the Student's t-distribution with $\nu=2.5$. The weighted error (orange) shows a clear $n^{-1/2}$ trend, unlike the uniform error (blue).*

---

##  Getting Started & Reproducibility

The simulation results presented in the paper are fully reproducible using the provided Python script.

### 1. Prerequisites
- Python 3.8+
- A Python virtual environment (recommended)

### 2. Installation
First, clone the repository:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name