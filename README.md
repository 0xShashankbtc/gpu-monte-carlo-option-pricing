# GPU Monte-Carlo Option Pricing and Portfolio Risk Analysis

A high-performance CUDA implementation for **Monte-Carlo European Option Pricing** and **Multi-Option Portfolio Risk Analysis (Value-at-Risk & Expected Shortfall)**, calibrated for National Stock Exchange (NSE) derivatives.

---

## 🚀 Key Features

1. **High-Throughput GPU Monte Carlo Engine**:
   - Generates independent standard normal random variables $Z \sim \mathcal{N}(0, 1)$ directly on GPU registers using NVIDIA **cuRAND Philox-4x32-10**.
   - Grid-stride loop design allows simulating tens to hundreds of millions of paths with minimal memory footprint.
   - Low-latency parallel reduction using warp shuffle intrinsics (`__shfl_down_sync`) and block-level shared memory reduction with 64-bit double-precision accumulators.
   - Delivers up to **3.0+ Billion simulations per second** on an NVIDIA GeForce RTX 3060 (12 GB).

2. **Rigorous Statistical & Analytical Validation**:
   - Computes empirical mean payoff and discounted option price $\hat{C} = e^{-rT} \frac{1}{N} \sum_{i=1}^N \text{Payoff}_i$.
   - Computes sample variance, standard error, and exact **95% and 99% Confidence Intervals**.
   - Validated against closed-form **Black-Scholes analytical formula** to analyze error convergence rates $\mathcal{O}(1/\sqrt{N})$.

3. **Extension: GPU Portfolio Risk & Value-at-Risk (VaR)**:
   - Models a real-world multi-asset, multi-option portfolio of NSE contracts (NIFTY 50 Index, RELIANCE, TCS, INFY, HDFCBANK).
   - Performs parallel revaluation across millions of market scenarios over an adjustable risk horizon (e.g., 10 trading days).
   - Computes **95% VaR**, **99% VaR**, and **Expected Shortfall (CVaR)** directly on the GPU using **CUDA Thrust** parallel radix sort and reduction.

4. **Multi-threaded CPU Benchmark**:
   - Includes a multi-threaded C++ reference implementation (`std::thread` + `std::mt19937_64`) to measure real-world GPU speedup.

---

## 🛠️ Project Structure

```text
├── src/
│   ├── common.h          # Data structures, CUDA checks, Black-Scholes analytical formulas
│   ├── option_mc.cu      # GPU Monte Carlo single-option pricing kernel + Warp-shuffle reduction
│   ├── portfolio_mc.cu   # GPU multi-option portfolio risk simulation & Thrust VaR/CVaR engine
│   ├── cpu_reference.h   # Multi-threaded CPU reference implementation
│   └── main.cu           # Main CLI driver and interactive benchmark coordinator
├── results/              # Exported JSON benchmark data and publication-quality plots
│   ├── convergence_analysis.png
│   ├── error_decay.png
│   ├── execution_time_throughput.png
│   ├── portfolio_var_distribution.png
│   └── benchmark_data.json
├── build.bat             # 1-Click Windows build script for NVCC & MSVC
├── plot_results.py       # Python script for generating publication figures
├── README.md             # Project user guide
└── PROJECT_REPORT.md     # Academic project report for submission
```

---

## ⚡ Quick Start

### Prerequisites
- **Operating System**: Windows 10/11 (or Linux)
- **NVIDIA GPU**: Compute Capability 7.0+ (Tested on RTX 3060, Compute Capability 8.6)
- **CUDA Toolkit**: CUDA 11.x, 12.x, or 13.x with `nvcc`
- **C++ Compiler**: Microsoft Visual Studio Build Tools (MSVC `cl.exe`) or GCC
- **Python (Optional for plots)**: Python 3.9+ with `numpy`, `matplotlib`

### 1. Build the Executable
Run the automated build script:
```cmd
build.bat
```
This builds `option_pricer.exe` with `-O3 -std=c++17 -arch=sm_86`.

### 2. Run Single Option Pricing & Portfolio Analysis
Run with default NSE parameters (NIFTY 50 Call: $S_0 = 24500$, $K = 24500$, $r = 6.5\%$, $\sigma = 18\%$, $T = 0.25$ years, $10,000,000$ simulations):
```cmd
.\option_pricer.exe
```

### 3. Custom CLI Options
```cmd
.\option_pricer.exe --S0 24500 --K 24500 --r 0.065 --sigma 0.18 --T 0.25 --sims 10000000 --type call --mode all --json results/benchmark_data.json
```

| Flag | Description | Default |
|---|---|---|
| `--S0 <val>` | Underlying asset spot price | `24500.0` (NIFTY 50) |
| `--K <val>` | Strike price | `24500.0` |
| `--r <val>` | Risk-free rate (annualized) | `0.065` (6.5%) |
| `--sigma <val>` | Volatility (annualized) | `0.18` (18%) |
| `--T <val>` | Time to expiration (years) | `0.25` (3 months) |
| `--sims <N>` | Number of Monte Carlo simulations | `10000000` (10M) |
| `--type <call\|put>` | Option contract type | `call` |
| `--mode <single\|sweep\|portfolio\|all>` | Execution mode | `all` |
| `--horizon <days>` | VaR risk horizon (trading days) | `10.0` |
| `--json <file>` | Path to export benchmark JSON | `results/benchmark_data.json` |

### 4. Generate Visualizations
To produce the graphical figures:
```cmd
python plot_results.py
```
Generated plots are placed in `results/`.
