import json
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    json_path = os.path.join("results", "benchmark_data.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs("results", exist_ok=True)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    sweep = data["convergence_sweep"]
    N_vals = np.array([item["N"] for item in sweep])
    gpu_prices = np.array([item["gpu_price"] for item in sweep])
    std_errors = np.array([item["std_error"] for item in sweep])
    bs_price = sweep[0]["bs_price"]

    ci95_lower = gpu_prices - 1.959964 * std_errors
    ci95_upper = gpu_prices + 1.959964 * std_errors

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(N_vals, gpu_prices, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='GPU Monte Carlo Estimate')
    ax.fill_between(N_vals, ci95_lower, ci95_upper, color='#1f77b4', alpha=0.2, label='95% Confidence Interval Band')
    ax.axhline(bs_price, color='#d62728', linestyle='--', linewidth=2, label=f'Black-Scholes Analytical (INR {bs_price:.2f})')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Monte-Carlo Simulations (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estimated Call Option Price (INR)', fontsize=12, fontweight='bold')
    ax.set_title('Monte-Carlo Price Convergence to Black-Scholes Analytical Benchmark\n(NSE NIFTY 50 Call: S0=24500, K=24500, r=6.5%, σ=18%, T=0.25y)', fontsize=13, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc='best', frameon=True, shadow=True)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "convergence_analysis.png"))
    plt.close(fig)
    print("Saved: results/convergence_analysis.png")

    abs_errors = np.array([item["abs_error"] for item in sweep])
    theoretical_decay = std_errors[0] * np.sqrt(N_vals[0]) / np.sqrt(N_vals)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.loglog(N_vals, std_errors, 's-', color='#2ca02c', linewidth=2, markersize=7, label=r'Monte Carlo Standard Error ($SE = \sigma / \sqrt{N}$)')
    ax.loglog(N_vals, abs_errors, 'd--', color='#ff7f0e', linewidth=1.8, markersize=7, label='Empirical Absolute Error (|Estimate - BS|)')
    ax.loglog(N_vals, theoretical_decay, 'k:', linewidth=2, label=r'Theoretical Rate $\mathcal{O}(1/\sqrt{N})$')

    ax.set_xlabel('Number of Simulations (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (INR)', fontsize=12, fontweight='bold')
    ax.set_title('Error Convergence Rate vs Theoretical Rate on Log-Log Scale\nVerification of Central Limit Theorem (CLT)', fontsize=13, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc='best', frameon=True, shadow=True)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "error_decay.png"))
    plt.close(fig)
    print("Saved: results/error_decay.png")

    gpu_times = np.array([item["gpu_time_ms"] for item in sweep])
    throughputs = np.array([item["throughput_m_sims"] for item in sweep])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ax1.plot(N_vals, gpu_times, 'o-', color='#9467bd', linewidth=2.2, markersize=7, label='GPU Total Time (RTX 3060)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Simulations (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('GPU Execution Time Scaling', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(loc='upper left', frameon=True)

    ax2.plot(N_vals, throughputs, 's-', color='#e377c2', linewidth=2.2, markersize=7, label='Throughput (Million sims/sec)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Simulations (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (Million Simulations / Sec)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Simulation Throughput (Saturating 28 SMs)', fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(loc='lower right', frameon=True)

    fig.suptitle('Hardware Performance on NVIDIA GeForce RTX 3060 (3,584 CUDA Cores)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join("results", "execution_time_throughput.png"))
    plt.close(fig)
    print("Saved: results/execution_time_throughput.png")

    port = data["portfolio_risk"]
    sample_losses = np.array(port["sample_losses"])
    init_val = port["initial_portfolio_value"]
    var95 = port["var_95"]
    var99 = port["var_99"]
    cvar95 = port["cvar_95"]
    cvar99 = port["cvar_99"]

    pnl = -sample_losses

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    counts, bins, patches = ax.hist(pnl, bins=120, density=True, color='#4a90e2', alpha=0.65, edgecolor='black', linewidth=0.5, label='Empirical Simulated P&L')

    pnl_var95 = -var95
    pnl_var99 = -var99

    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < pnl_var99:
            patch.set_facecolor('#d9534f')
        elif left_edge < pnl_var95:
            patch.set_facecolor('#f0ad4e')

    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, label='Breakeven (P&L = 0)')
    ax.axvline(pnl_var95, color='#e67e22', linestyle='--', linewidth=2.2, label=f'95% VaR: -INR {var95:.2f} (-{var95/init_val*100:.1f}%)')
    ax.axvline(pnl_var99, color='#c0392b', linestyle='-', linewidth=2.2, label=f'99% VaR: -INR {var99:.2f} (-{var99/init_val*100:.1f}%)')
    ax.axvline(-cvar95, color='#8e44ad', linestyle=':', linewidth=2.0, label=f'95% CVaR (ES): -INR {cvar95:.2f}')
    ax.axvline(-cvar99, color='#2c3e50', linestyle='-.', linewidth=2.0, label=f'99% CVaR (ES): -INR {cvar99:.2f}')

    ax.set_xlabel('Simulated Portfolio Profit & Loss (INR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title(f'GPU Multi-Option Portfolio Risk Analysis (5,000,000 Monte Carlo Paths)\nRisk Horizon: 10 Trading Days | Initial Portfolio Value: INR {init_val:,.2f}', fontsize=13, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)

    fig.tight_layout()
    fig.savefig(os.path.join("results", "portfolio_var_distribution.png"))
    plt.close(fig)
    print("Saved: results/portfolio_var_distribution.png")

    print("\nAll 4 publication plots generated successfully in results/ directory!")

if __name__ == '__main__':
    main()
