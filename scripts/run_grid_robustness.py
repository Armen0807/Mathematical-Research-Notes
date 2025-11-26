import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def weighted_kolmogorov_metric(data, dist, q, h_func=None):
    """
    Computes the Weighted Kolmogorov Distance d_{K,h,q}.

    Args:
        data: Array of empirical returns.
        dist: Theoretical distribution (scipy object with .cdf).
        q: Weight parameter.
        h_func: Exhaustion function. Defaults to |x|.
    """
    n = len(data)
    sorted_data = np.sort(data)

    y_emp = np.arange(1, n + 1) / n
    y_theo = dist.cdf(sorted_data)


    error_plus = np.abs(y_emp - y_theo)
    error_minus = np.abs(np.arange(0, n) / n - y_theo)
    base_errors = np.maximum(error_plus, error_minus)



    if h_func is None:
        h_x = np.abs(sorted_data)
    else:
        h_x = h_func(sorted_data)


    weights = (1 + h_x) ** (-q)


    d_weighted = np.max(base_errors * weights)

    return d_weighted


def grid_robustness_test(data, model_dist, q_min=0.5, q_max=2.5, n_steps=20, threshold=0.05):
    """
    Implements the Anti-Gaming Protocol (Section 8.2).
    Rejects the model if it fails ANYWHERE on the q-grid.
    """
    q_grid = np.linspace(q_min, q_max, n_steps)
    scores = []

    for q in q_grid:
        score = weighted_kolmogorov_metric(data, model_dist, q)
        scores.append(score)

    scores = np.array(scores)

    d_rob = np.max(scores)

    passed = d_rob <= threshold

    return q_grid, scores, passed, d_rob



if __name__ == "__main__":
    np.random.seed(42)


    n_samples = 1000
    market_data = stats.t.rvs(df=3, size=n_samples)


    mu, std = stats.norm.fit(market_data)
    model_gaussian = stats.norm(loc=mu, scale=std)

    params = stats.t.fit(market_data)
    model_student = stats.t(df=params[0], loc=params[1], scale=params[2])


    epsilon_core = 0.04

    q_vals, scores_gauss, pass_gauss, max_gauss = grid_robustness_test(market_data, model_gaussian,
                                                                       threshold=epsilon_core)
    q_vals, scores_stud, pass_stud, max_stud = grid_robustness_test(market_data, model_student, threshold=epsilon_core)

    plt.figure(figsize=(10, 6))

    plt.axhline(y=epsilon_core, color='black', linestyle='--', linewidth=2,
                label=f'Rejection Threshold ($\\epsilon_{{core}}={epsilon_core}$)')

    plt.plot(q_vals, scores_gauss, marker='o', color='red', label=f'Gaussian Model (Max Error: {max_gauss:.4f})')

    plt.plot(q_vals, scores_stud, marker='o', color='green', label=f'Student-t Model (Max Error: {max_stud:.4f})')

    plt.fill_between(q_vals, 0, epsilon_core, color='gray', alpha=0.1)

    plt.title("Grid Robustness Analysis (Anti-Gaming Protocol)", fontsize=14)
    plt.xlabel("Weight Parameter $q$ (Tail Penalty)", fontsize=12)
    plt.ylabel("Weighted Kolmogorov Error $d_{K,h,q}$", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    status_g = "REJECTED" if not pass_gauss else "ACCEPTED"
    status_s = "REJECTED" if not pass_stud else "ACCEPTED"

    print(f"Gaussian Model: {status_g} (Worst-case metric: {max_gauss:.4f})")
    print(f"Student Model:  {status_s} (Worst-case metric: {max_stud:.4f})")

    plt.show()