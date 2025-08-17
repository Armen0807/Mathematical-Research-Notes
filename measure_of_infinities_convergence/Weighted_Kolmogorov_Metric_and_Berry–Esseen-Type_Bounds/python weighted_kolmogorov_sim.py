# weighted_kolmogorov_sim.py
# Generate weighted vs uniform Kolmogorov errors for Student(ν) and Pareto(α)
# Outputs: CSV + PNGs in ./out
# No seaborn. Pure matplotlib. Single-plot figures. No explicit colors.

import argparse
import os
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf as m_erf


_erf_vec = np.vectorize(lambda z: m_erf(float(z)))

def Phi(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf_vec(x / np.sqrt(2.0)))


# ---------- Weighted & uniform Kolmogorov errors from ECDF ----------
def weighted_and_uniform_errors(Z_samples: np.ndarray, q: float, t_grid: np.ndarray):
    """
    Z_samples: shape (B,), i.i.d. draws of Z_n (normalized sum) for fixed n
    t_grid   : evaluation grid in R (1D array)
    Returns: (weighted_error, uniform_error)
    """
    Z_sorted = np.sort(Z_samples)
    counts = np.searchsorted(Z_sorted, t_grid, side="right")
    F_emp = counts / Z_sorted.size
    w = 1.0 / (1.0 + np.abs(t_grid)) ** q
    diff = np.abs(F_emp - Phi(t_grid))
    return np.max(w * diff), np.max(diff)

# ---------- Student-t model ----------
def simulate_student(nu: float, n_values, B: int, q: float,
                     tmin=-8.0, tmax=8.0, grid_points=4001, seed=123):
    rng = np.random.default_rng(seed)
    # variance exists for nu>2
    sigma = sqrt(nu / (nu - 2.0))
    t_grid = np.linspace(tmin, tmax, grid_points)
    w_errs, u_errs = [], []
    for n in n_values:
        # Simulate B sums of size n
        X = rng.standard_t(df=nu, size=(B, n))  # mean 0
        S = X.sum(axis=1)
        Z = S / (sigma * np.sqrt(n))
        we, ue = weighted_and_uniform_errors(Z, q=q, t_grid=t_grid)
        w_errs.append(we)
        u_errs.append(ue)
    return np.array(w_errs), np.array(u_errs)

# ---------- Pareto model (Type I, xm=1) centered ----------
def simulate_pareto(alpha: float, n_values, B: int, q: float,
                    tmin=-8.0, tmax=8.0, grid_points=4001, seed=456):
    rng = np.random.default_rng(seed)
    # Type I Pareto with xm=1 has mean mu=alpha/(alpha-1) (alpha>1)
    # variance = alpha/((alpha-1)^2 (alpha-2)) for alpha>2
    mu = alpha / (alpha - 1.0)
    var = alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0))
    sigma = sqrt(var)
    t_grid = np.linspace(tmin, tmax, grid_points)
    w_errs, u_errs = [], []
    for n in n_values:
        # numpy pareto(alpha) has support y>=0 with pdf a*(1+y)^(-a-1); X=Y+1 is Type I with xm=1
        Y = rng.pareto(alpha, size=(B, n)) + 1.0
        X = Y - mu  # center to mean 0
        S = X.sum(axis=1)
        Z = S / (sigma * np.sqrt(n))
        we, ue = weighted_and_uniform_errors(Z, q=q, t_grid=t_grid)
        w_errs.append(we)
        u_errs.append(ue)
    return np.array(w_errs), np.array(u_errs)

# ---------- Plot helpers ----------
def ensure_outdir(path="out"):
    os.makedirs(path, exist_ok=True)
    return path

def plot_loglog(x, y, title, xlabel, ylabel, outfile, ref_slope=None):
    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, marker="o")
    if ref_slope is not None:
        # draw a reference line of slope `ref_slope` through the last point for visual guidance
        x0, y0 = x[-1], y[-1]
        y_ref = y0 * (x / x0) ** ref_slope
        plt.loglog(x, y_ref, linestyle="--", label=f"reference slope {ref_slope:g}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":")
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close()

def plot_compare(x, y1, y2, title, xlabel, ylabel, labels, outfile, ref_slope=None):
    plt.figure(figsize=(7, 5))
    plt.loglog(x, y1, marker="s", label=labels[0])
    plt.loglog(x, y2, marker="o", label=labels[1])
    if ref_slope is not None:
        x0, y0 = x[-1], y2[-1]
        y_ref = y0 * (x / x0) ** ref_slope
        plt.loglog(x, y_ref, linestyle="--", label=f"reference slope {ref_slope:g}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Weighted vs uniform Kolmogorov error for heavy-tailed sums")
    parser.add_argument("--nu", type=float, default=2.5, help="Student t degrees of freedom (nu>2)")
    parser.add_argument("--alpha", type=float, default=2.8, help="Pareto alpha (>2)")
    parser.add_argument("--q", type=float, default=1.2, help="weight exponent in w_q(t)=(1+|t|)^(-q)")
    parser.add_argument("--B", type=int, default=160, help="number of sums per n")
    parser.add_argument("--n_min_exp", type=int, default=8, help="min exponent for n=2^k")
    parser.add_argument("--n_max_exp", type=int, default=15, help="max exponent for n=2^k")
    parser.add_argument("--grid_min", type=float, default=-8.0, help="grid lower bound")
    parser.add_argument("--grid_max", type=float, default=8.0, help="grid upper bound")
    parser.add_argument("--grid_points", type=int, default=4001, help="grid resolution")
    parser.add_argument("--outdir", type=str, default="out", help="output directory")
    args = parser.parse_args()

    n_values = np.array([2 ** k for k in range(args.n_min_exp, args.n_max_exp + 1)], dtype=int)
    outdir = ensure_outdir(args.outdir)

    # Simulations
    w_stu, u_stu = simulate_student(
        nu=args.nu, n_values=n_values, B=args.B, q=args.q,
        tmin=args.grid_min, tmax=args.grid_max, grid_points=args.grid_points, seed=123
    )
    w_par, u_par = simulate_pareto(
        alpha=args.alpha, n_values=n_values, B=args.B, q=args.q,
        tmin=args.grid_min, tmax=args.grid_max, grid_points=args.grid_points, seed=456
    )

    # Save CSV
    df = pd.DataFrame({
        "n": n_values,
        "weighted_student": w_stu,
        "uniform_student": u_stu,
        "weighted_pareto": w_par,
        "uniform_pareto": u_par
    })
    csv_path = os.path.join(outdir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Figures
    fig1 = os.path.join(outdir, "weighted_student.png")
    plot_loglog(
        n_values, w_stu,
        title=f"Student t(ν={args.nu}): weighted Kolmogorov error vs n (q={args.q})",
        xlabel="n", ylabel="Weighted Kolmogorov error", outfile=fig1, ref_slope=-0.5
    )
    print(f"[saved] {fig1}")

    fig2 = os.path.join(outdir, "compare_student.png")
    plot_compare(
        n_values, u_stu, w_stu,
        title=f"Student t(ν={args.nu}): uniform vs weighted errors (q={args.q})",
        xlabel="n", ylabel="Error",
        labels=("Uniform Kolmogorov", "Weighted"),
        outfile=fig2, ref_slope=-0.5
    )
    print(f"[saved] {fig2}")

    fig3 = os.path.join(outdir, "weighted_pareto.png")
    plot_loglog(
        n_values, w_par,
        title=f"Pareto(α={args.alpha}): weighted Kolmogorov error vs n (q={args.q})",
        xlabel="n", ylabel="Weighted Kolmogorov error", outfile=fig3, ref_slope=-0.5
    )
    print(f"[saved] {fig3}")

if __name__ == "__main__":
    main()
