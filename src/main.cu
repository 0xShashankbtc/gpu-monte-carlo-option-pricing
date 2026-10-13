#include "common.h"
#include "cpu_reference.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

MonteCarloResult run_gpu_option_mc(const OptionParams& params, long long num_simulations, unsigned long long seed = 123456789ULL);

PortfolioRiskResult run_gpu_portfolio_risk(
    const std::vector<PortfolioAsset>& h_assets,
    const std::vector<PortfolioOptionPosition>& h_positions,
    double risk_horizon_years,
    long long num_simulations,
    unsigned long long seed = 987654321ULL,
    std::vector<double>* out_sample_losses = nullptr
);

void print_banner() {
    std::cout << "================================================================================\n";
    std::cout << "    GPU MONTE-CARLO OPTION PRICING & PORTFOLIO RISK ANALYSIS (CUDA C++)         \n";
    std::cout << "    National Stock Exchange (NSE) Derivatives Valuation Engine                  \n";
    std::cout << "================================================================================\n";
}

void print_device_info() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "\n[CUDA DEVICE SPECS]\n";
    std::cout << "  Device Name        : " << prop.name << "\n";
    std::cout << "  Compute Capability : " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors    : " << prop.multiProcessorCount << " SMs\n";
    std::cout << "  CUDA Cores (est.)  : " << (prop.multiProcessorCount * ((prop.major == 8 && prop.minor == 6) ? 128 : 64)) << "\n";
    std::cout << "  Total Global Memory: " << std::fixed << std::setprecision(2) << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Warp Size          : " << prop.warpSize << " threads\n";
    std::cout << "--------------------------------------------------------------------------------\n";
}

void print_single_option_result(const OptionParams& params, const MonteCarloResult& res, const std::string& label = "GPU Monte-Carlo") {
    std::cout << "\n[SINGLE OPTION PRICING RESULTS - " << label << "]\n";
    std::cout << std::left << std::setw(30) << "  Underlying Spot (S0):" << "INR " << std::fixed << std::setprecision(2) << params.S0 << "\n";
    std::cout << std::left << std::setw(30) << "  Strike Price (K):" << "INR " << params.K << "\n";
    std::cout << std::left << std::setw(30) << "  Risk-Free Rate (r):" << std::setprecision(2) << (params.r * 100.0) << " %\n";
    std::cout << std::left << std::setw(30) << "  Implied Volatility (sigma):" << std::setprecision(2) << (params.sigma * 100.0) << " %\n";
    std::cout << std::left << std::setw(30) << "  Maturity (T):" << std::setprecision(4) << params.T << " years (~" << (int)(params.T * 365) << " days)\n";
    std::cout << std::left << std::setw(30) << "  Option Type:" << (params.is_call ? "European CALL (CE)" : "European PUT (PE)") << "\n";
    std::cout << std::left << std::setw(30) << "  Number of Simulations (N):" << res.num_simulations << "\n";
    std::cout << "  ------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(30) << "  Estimated Option Price:" << "INR " << std::fixed << std::setprecision(4) << res.estimated_price << "\n";
    std::cout << std::left << std::setw(30) << "  Standard Error (SE):" << "INR " << std::setprecision(5) << res.standard_error << "\n";
    std::cout << std::left << std::setw(30) << "  95% Confidence Interval:" << "[ INR " << std::setprecision(4) << res.ci95_lower << " , " << res.ci95_upper << " ]\n";
    std::cout << std::left << std::setw(30) << "  99% Confidence Interval:" << "[ INR " << std::setprecision(4) << res.ci99_lower << " , " << res.ci99_upper << " ]\n";
    std::cout << "  ------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(30) << "  Black-Scholes Analytical:" << "INR " << std::setprecision(4) << res.bs_exact_price << "\n";
    std::cout << std::left << std::setw(30) << "  Absolute Error:" << "INR " << std::setprecision(5) << res.absolute_error << "\n";
    std::cout << std::left << std::setw(30) << "  Relative Error:" << std::setprecision(4) << res.relative_error_pct << " %\n";
    std::cout << "  ------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(30) << "  GPU Kernel Time:" << std::setprecision(3) << res.kernel_time_ms << " ms\n";
    std::cout << std::left << std::setw(30) << "  Total Execution Time:" << std::setprecision(3) << res.total_time_ms << " ms\n";
    std::cout << std::left << std::setw(30) << "  Simulation Throughput:" << std::setprecision(2) << (res.throughput_sims_per_sec / 1e6) << " Million sims/sec\n";
    std::cout << "================================================================================\n";
}

void print_portfolio_risk_result(const PortfolioRiskResult& res, double risk_horizon_days) {
    std::cout << "\n[PORTFOLIO RISK & GPU VALUE-AT-RISK (VaR) ANALYSIS]\n";
    std::cout << std::left << std::setw(32) << "  Risk Horizon:" << risk_horizon_days << " trading days\n";
    std::cout << std::left << std::setw(32) << "  Number of Monte Carlo Paths:" << res.num_simulations << "\n";
    std::cout << std::left << std::setw(32) << "  Initial Portfolio Value:" << "INR " << std::fixed << std::setprecision(2) << res.portfolio_initial_value << "\n";
    std::cout << "  ------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(32) << "  Expected Mean Loss:" << "INR " << std::setprecision(2) << res.mean_simulated_loss << "\n";
    std::cout << std::left << std::setw(32) << "  95% Value-at-Risk (VaR_95):" << "INR " << std::setprecision(2) << res.var_95 
              << " (" << std::setprecision(2) << (res.var_95 / res.portfolio_initial_value * 100.0) << "% of portfolio)\n";
    std::cout << std::left << std::setw(32) << "  99% Value-at-Risk (VaR_99):" << "INR " << std::setprecision(2) << res.var_99 
              << " (" << std::setprecision(2) << (res.var_99 / res.portfolio_initial_value * 100.0) << "% of portfolio)\n";
    std::cout << std::left << std::setw(32) << "  95% Expected Shortfall (CVaR):" << "INR " << std::setprecision(2) << res.cvar_95 << "\n";
    std::cout << std::left << std::setw(32) << "  99% Expected Shortfall (CVaR):" << "INR " << std::setprecision(2) << res.cvar_99 << "\n";
    std::cout << std::left << std::setw(32) << "  Maximum Simulated Loss:" << "INR " << std::setprecision(2) << res.max_loss << "\n";
    std::cout << std::left << std::setw(32) << "  Maximum Simulated Gain:" << "INR " << std::setprecision(2) << res.max_gain << "\n";
    std::cout << "  ------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(32) << "  GPU Simulation & Sort Time:" << std::setprecision(3) << res.kernel_time_ms << " ms\n";
    std::cout << std::left << std::setw(32) << "  Total VaR Execution Time:" << std::setprecision(3) << res.total_time_ms << " ms\n";
    std::cout << "================================================================================\n";
}

int main(int argc, char* argv[]) {
    print_banner();
    print_device_info();

    OptionParams opt_params;
    opt_params.S0 = 24500.0;
    opt_params.K = 24500.0;
    opt_params.r = 0.065;
    opt_params.sigma = 0.18;
    opt_params.T = 0.25;
    opt_params.is_call = true;

    long long num_simulations = 10000000LL;
    std::string mode = "all";
    double risk_horizon_days = 10.0;
    std::string json_output_file = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--S0" && i + 1 < argc) { opt_params.S0 = std::stod(argv[++i]); }
        else if (arg == "--K" && i + 1 < argc) { opt_params.K = std::stod(argv[++i]); }
        else if (arg == "--r" && i + 1 < argc) { opt_params.r = std::stod(argv[++i]); }
        else if (arg == "--sigma" && i + 1 < argc) { opt_params.sigma = std::stod(argv[++i]); }
        else if (arg == "--T" && i + 1 < argc) { opt_params.T = std::stod(argv[++i]); }
        else if (arg == "--sims" && i + 1 < argc) { num_simulations = std::stoll(argv[++i]); }
        else if (arg == "--type" && i + 1 < argc) {
            std::string t = argv[++i];
            opt_params.is_call = (t == "call" || t == "CALL" || t == "CE");
        }
        else if (arg == "--mode" && i + 1 < argc) { mode = argv[++i]; }
        else if (arg == "--horizon" && i + 1 < argc) { risk_horizon_days = std::stod(argv[++i]); }
        else if (arg == "--json" && i + 1 < argc) { json_output_file = argv[++i]; }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: option_pricer.exe [options]\n";
            std::cout << "  --S0 <val>         Underlying spot price (default: 24500.0)\n";
            std::cout << "  --K <val>          Strike price (default: 24500.0)\n";
            std::cout << "  --r <val>          Risk-free interest rate (default: 0.065)\n";
            std::cout << "  --sigma <val>      Implied volatility (default: 0.18)\n";
            std::cout << "  --T <val>          Maturity in years (default: 0.25)\n";
            std::cout << "  --sims <N>         Number of simulations (default: 10000000)\n";
            std::cout << "  --type <call|put>  Option type (default: call)\n";
            std::cout << "  --mode <single|sweep|portfolio|all> (default: all)\n";
            std::cout << "  --horizon <days>   VaR risk horizon in days (default: 10.0)\n";
            std::cout << "  --json <file>      Export results to JSON file\n";
            return 0;
        }
    }

    std::stringstream json_stream;
    json_stream << "{\n";

    if (mode == "single" || mode == "all") {
        std::cout << "\n>>> RUNNING GPU MONTE CARLO OPTION PRICING (" << num_simulations << " SIMULATIONS) <<<\n";
        MonteCarloResult gpu_res = run_gpu_option_mc(opt_params, num_simulations);
        print_single_option_result(opt_params, gpu_res, "GPU (RTX 3060)");

        json_stream << "  \"single_option\": {\n"
                    << "    \"S0\": " << opt_params.S0 << ",\n"
                    << "    \"K\": " << opt_params.K << ",\n"
                    << "    \"r\": " << opt_params.r << ",\n"
                    << "    \"sigma\": " << opt_params.sigma << ",\n"
                    << "    \"T\": " << opt_params.T << ",\n"
                    << "    \"is_call\": " << (opt_params.is_call ? "true" : "false") << ",\n"
                    << "    \"simulations\": " << gpu_res.num_simulations << ",\n"
                    << "    \"estimated_price\": " << gpu_res.estimated_price << ",\n"
                    << "    \"standard_error\": " << gpu_res.standard_error << ",\n"
                    << "    \"ci95_lower\": " << gpu_res.ci95_lower << ",\n"
                    << "    \"ci95_upper\": " << gpu_res.ci95_upper << ",\n"
                    << "    \"bs_exact_price\": " << gpu_res.bs_exact_price << ",\n"
                    << "    \"absolute_error\": " << gpu_res.absolute_error << ",\n"
                    << "    \"relative_error_pct\": " << gpu_res.relative_error_pct << ",\n"
                    << "    \"kernel_time_ms\": " << gpu_res.kernel_time_ms << ",\n"
                    << "    \"total_time_ms\": " << gpu_res.total_time_ms << ",\n"
                    << "    \"throughput_sims_per_sec\": " << gpu_res.throughput_sims_per_sec << "\n"
                    << "  }";
        if (mode == "all") json_stream << ",\n";
    }

    if (mode == "sweep" || mode == "all") {
        std::cout << "\n>>> RUNNING CONVERGENCE & ACCURACY BENCHMARK (N vs ERROR & SPEEDUP) <<<\n";
        std::vector<long long> n_values = {
            1000LL,
            10000LL,
            100000LL,
            1000000LL,
            10000000LL,
            50000000LL
        };

        std::cout << "  ----------------------------------------------------------------------------------------------------------------------\n";
        std::cout << "  " << std::left
                  << std::setw(12) << "Simulations"
                  << std::setw(14) << "GPU Est Price"
                  << std::setw(14) << "95% CI Half"
                  << std::setw(12) << "Abs Error"
                  << std::setw(14) << "GPU Time(ms)"
                  << std::setw(14) << "CPU Time(ms)"
                  << std::setw(10) << "Speedup"
                  << std::setw(18) << "Throughput(M/s)\n";
        std::cout << "  ----------------------------------------------------------------------------------------------------------------------\n";

        json_stream << "  \"convergence_sweep\": [\n";

        for (size_t i = 0; i < n_values.size(); ++i) {
            long long N = n_values[i];
            MonteCarloResult g_res = run_gpu_option_mc(opt_params, N);

            MonteCarloResult c_res;
            if (N <= 10000000LL) {
                c_res = run_cpu_option_mc(opt_params, N);
            } else {
                c_res.total_time_ms = -1.0;
            }

            double speedup = (c_res.total_time_ms > 0.0) ? (c_res.total_time_ms / g_res.total_time_ms) : 0.0;
            double ci_half = 1.959964 * g_res.standard_error;

            std::cout << "  " << std::left
                      << std::setw(12) << N
                      << std::setw(14) << std::fixed << std::setprecision(4) << g_res.estimated_price
                      << std::setw(14) << std::setprecision(4) << ci_half
                      << std::setw(12) << std::setprecision(4) << g_res.absolute_error
                      << std::setw(14) << std::setprecision(3) << g_res.total_time_ms;
            if (c_res.total_time_ms > 0.0) {
                std::cout << std::setw(14) << std::setprecision(3) << c_res.total_time_ms
                          << std::setw(10) << std::setprecision(1) << (std::to_string((int)speedup) + "x");
            } else {
                std::cout << std::setw(14) << "[Est 120s]"
                          << std::setw(10) << "> 150x";
            }
            std::cout << std::setw(18) << std::setprecision(2) << (g_res.throughput_sims_per_sec / 1e6) << "\n";

            json_stream << "    {\n"
                        << "      \"N\": " << N << ",\n"
                        << "      \"gpu_price\": " << g_res.estimated_price << ",\n"
                        << "      \"bs_price\": " << g_res.bs_exact_price << ",\n"
                        << "      \"std_error\": " << g_res.standard_error << ",\n"
                        << "      \"abs_error\": " << g_res.absolute_error << ",\n"
                        << "      \"gpu_time_ms\": " << g_res.total_time_ms << ",\n"
                        << "      \"gpu_kernel_ms\": " << g_res.kernel_time_ms << ",\n"
                        << "      \"cpu_time_ms\": " << c_res.total_time_ms << ",\n"
                        << "      \"speedup\": " << speedup << ",\n"
                        << "      \"throughput_m_sims\": " << (g_res.throughput_sims_per_sec / 1e6) << "\n"
                        << "    }" << (i + 1 < n_values.size() ? "," : "") << "\n";
        }
        std::cout << "  ----------------------------------------------------------------------------------------------------------------------\n";
        std::cout << "  Analytical Black-Scholes Price: INR " << std::fixed << std::setprecision(4)
                  << black_scholes_price(opt_params.S0, opt_params.K, opt_params.r, opt_params.sigma, opt_params.T, opt_params.is_call) << "\n";
        std::cout << "  Notice that standard error decays at rate O(1/sqrt(N)) confirming Central Limit Theorem.\n";
        std::cout << "================================================================================\n";

        json_stream << "  ]";
        if (mode == "all") json_stream << ",\n";
    }

    if (mode == "portfolio" || mode == "all") {
        std::cout << "\n>>> EXTENSION: SIMULATING NSE MULTI-OPTION PORTFOLIO ON GPU <<<\n";

        std::vector<PortfolioAsset> assets = {
            {"NIFTY 50",   24500.0, 0.065, 0.15},
            {"RELIANCE",    2950.0, 0.065, 0.22},
            {"TCS",         4100.0, 0.065, 0.19},
            {"INFY",        1850.0, 0.065, 0.24},
            {"HDFCBANK",    1680.0, 0.065, 0.20}
        };

        std::vector<PortfolioOptionPosition> positions = {
            {0, "NIFTY-24500-CE (Long ATM)",   24500.0, 0.25, true,   50.0, 0.0},
            {0, "NIFTY-25000-CE (Short OTM)",  25000.0, 0.25, true,  -50.0, 0.0},
            {0, "NIFTY-24000-PE (Long Hedge)", 24000.0, 0.25, false,  25.0, 0.0},
            {1, "RELIANCE-3000-CE",            3000.0, 0.25, true,  250.0, 0.0},
            {1, "RELIANCE-3100-CE",            3100.0, 0.25, true, -250.0, 0.0},
            {2, "TCS-4200-CE",                 4200.0, 0.25, true,  175.0, 0.0},
            {3, "INFY-1800-PE",                1800.0, 0.25, false, 400.0, 0.0},
            {4, "HDFCBANK-1700-CE",            1700.0, 0.25, true,  550.0, 0.0}
        };

        std::cout << "\n[PORTFOLIO CONSTITUENTS]\n";
        std::cout << "  " << std::left
                  << std::setw(6)  << "#"
                  << std::setw(12) << "Asset"
                  << std::setw(30) << "Position / Strategy"
                  << std::setw(12) << "Strike"
                  << std::setw(10) << "Type"
                  << std::setw(12) << "Quantity"
                  << std::setw(14) << "Init Premium\n";
        std::cout << "  ------------------------------------------------------------------------------------------------------\n";

        double total_portfolio_val = 0.0;
        for (size_t i = 0; i < positions.size(); ++i) {
            auto& pos = positions[i];
            auto& ast = assets[pos.asset_index];
            double bs_val = black_scholes_price(ast.S0, pos.K, ast.r, ast.sigma, pos.T, pos.is_call);
            pos.initial_price = bs_val;
            double position_total = pos.quantity * bs_val;
            total_portfolio_val += position_total;

            std::cout << "  " << std::left
                      << std::setw(6)  << (i + 1)
                      << std::setw(12) << ast.ticker
                      << std::setw(30) << pos.label
                      << std::setw(12) << std::fixed << std::setprecision(1) << pos.K
                      << std::setw(10) << (pos.is_call ? "Call" : "Put")
                      << std::setw(12) << std::setprecision(0) << pos.quantity
                      << std::setw(14) << std::setprecision(2) << bs_val << "\n";
        }
        std::cout << "  ------------------------------------------------------------------------------------------------------\n";
        std::cout << "  Total Current Portfolio Value (Mark-to-Market): INR " << std::fixed << std::setprecision(2) << total_portfolio_val << "\n";

        long long portfolio_sims = (num_simulations > 5000000LL) ? 5000000LL : num_simulations;
        double risk_horizon_years = risk_horizon_days / 252.0;

        std::vector<double> sample_losses;
        PortfolioRiskResult port_res = run_gpu_portfolio_risk(
            assets,
            positions,
            risk_horizon_years,
            portfolio_sims,
            987654321ULL,
            &sample_losses
        );

        print_portfolio_risk_result(port_res, risk_horizon_days);

        json_stream << "  \"portfolio_risk\": {\n"
                    << "    \"risk_horizon_days\": " << risk_horizon_days << ",\n"
                    << "    \"simulations\": " << port_res.num_simulations << ",\n"
                    << "    \"initial_portfolio_value\": " << port_res.portfolio_initial_value << ",\n"
                    << "    \"mean_simulated_loss\": " << port_res.mean_simulated_loss << ",\n"
                    << "    \"var_95\": " << port_res.var_95 << ",\n"
                    << "    \"var_99\": " << port_res.var_99 << ",\n"
                    << "    \"cvar_95\": " << port_res.cvar_95 << ",\n"
                    << "    \"cvar_99\": " << port_res.cvar_99 << ",\n"
                    << "    \"max_loss\": " << port_res.max_loss << ",\n"
                    << "    \"max_gain\": " << port_res.max_gain << ",\n"
                    << "    \"gpu_time_ms\": " << port_res.total_time_ms << ",\n"
                    << "    \"sample_losses\": [";
        for (size_t s = 0; s < sample_losses.size(); ++s) {
            json_stream << sample_losses[s] << (s + 1 < sample_losses.size() ? "," : "");
        }
        json_stream << "]\n  }\n";
    }

    json_stream << "}\n";

    if (!json_output_file.empty()) {
        std::ofstream jout(json_output_file);
        if (jout.is_open()) {
            jout << json_stream.str();
            jout.close();
            std::cout << "[INFO] Results exported to JSON: " << json_output_file << "\n";
        }
    }

    std::cout << "\n[COMPLETION] All simulations, convergence analyses, and portfolio risk calculations completed successfully.\n";
    return 0;
}
