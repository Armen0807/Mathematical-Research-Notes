import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm  # Barre de progression pro

# Importation depuis TON package (ça fait très pro)
from exhaustion_framework.metric import weighted_kolmogorov_error


def simulate_process(
        dist_type: str,
        params: dict,
        n_values: np.ndarray,
        B: int,
        q: float,
        t_grid: np.ndarray,
        seed: int
):
    rng = np.random.default_rng(seed)
    w_errs, u_errs = [], []

    iterator = tqdm(n_values, desc=f"Simulating {dist_type}", leave=True)

    for n in iterator:
        if dist_type == "student":
            nu = params['nu']
            sigma = sqrt(nu / (nu - 2.0))
            X = rng.standard_t(df=nu, size=(B, n))
            S = X.sum(axis=1)
            Z = S / (sigma * np.sqrt(n))

        elif dist_type == "pareto":
            alpha = params['alpha']
            mu = alpha / (alpha - 1.0)
            var = alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0))
            sigma = sqrt(var)
            Y = rng.pareto(alpha, size=(B, n)) + 1.0
            X = Y - mu
            S = X.sum(axis=1)
            Z = S / (sigma * np.sqrt(n))

        we, ue = weighted_kolmogorov_error(Z, q=q, t_grid=t_grid)
        w_errs.append(we)
        u_errs.append(ue)

    return np.array(w_errs), np.array(u_errs)


def plot_results(n_values, y_weighted, y_uniform, title, filename, outdir):
    """Helper function for clean plotting style."""
    plt.figure(figsize=(8, 6))
    # Style plus "scientifique"
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    plt.loglog(n_values, y_uniform, 's-', label="Uniform Kolmogorov", markersize=5, alpha=0.8)
    plt.loglog(n_values, y_weighted, 'o-', label="Weighted Metric (Ours)", markersize=5, linewidth=2)

    # Reference slope -0.5
    x0, y0 = n_values[-1], y_weighted[-1]
    y_ref = y0 * (n_values / x0) ** (-0.5)
    plt.loglog(n_values, y_ref, 'k--', label=r"Reference $O(n^{-1/2})$", alpha=0.6)

    plt.xlabel(r"Sample size $n$")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)

    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Reproduce Petrosyan (2025) Simulation Results")
    parser.add_argument("--nu", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=2.8)
    parser.add_argument("--q", type=float, default=1.2)
    parser.add_argument("--B", type=int, default=160, help="Batches per n")
    parser.add_argument("--outdir", type=str, default="experiments/results")

    args = parser.parse_args()

    n_values = np.array([2 ** k for k in range(8, 16)], dtype=int)
    t_grid = np.linspace(-8.0, 8.0, 4001)
    os.makedirs(args.outdir, exist_ok=True)

    w_stu, u_stu = simulate_process(
        "student", {"nu": args.nu}, n_values, args.B, args.q, t_grid, seed=123
    )

    w_par, u_par = simulate_process(
        "pareto", {"alpha": args.alpha}, n_values, args.B, args.q, t_grid, seed=456
    )

    df = pd.DataFrame({
        "n": n_values,
        "student_weighted": w_stu, "student_uniform": u_stu,
        "pareto_weighted": w_par, "pareto_uniform": u_par
    })
    df.to_csv(os.path.join(args.outdir, "convergence_results.csv"), index=False)

    plot_results(n_values, w_stu, u_stu,
                 f"Student-t (v={args.nu}) Convergence", "student_convergence.png", args.outdir)
    plot_results(n_values, w_par, u_par,
                 f"Pareto (a={args.alpha}) Convergence", "pareto_convergence.png", args.outdir)

    print(f"\n[SUCCESS] Results and plots generated in {args.outdir}")


if __name__ == "__main__":
    main()