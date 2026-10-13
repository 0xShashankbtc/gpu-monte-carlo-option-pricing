#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CURAND_CHECK(call)                                                    \
    do {                                                                      \
        curandStatus_t err = call;                                            \
        if (err != CURAND_STATUS_SUCCESS) {                                   \
            std::cerr << "CURAND Error at " << __FILE__ << ":" << __LINE__   \
                      << " -> code " << err << std::endl;                     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

struct OptionParams {
    double S0;
    double K;
    double r;
    double sigma;
    double T;
    bool is_call;
};

struct MonteCarloResult {
    double estimated_price;
    double standard_error;
    double ci95_lower;
    double ci95_upper;
    double ci99_lower;
    double ci99_upper;
    double bs_exact_price;
    double absolute_error;
    double relative_error_pct;
    double kernel_time_ms;
    double total_time_ms;
    double throughput_sims_per_sec;
    long long num_simulations;
};

struct PortfolioAsset {
    char ticker[32];
    double S0;
    double r;
    double sigma;
};

struct PortfolioOptionPosition {
    int asset_index;
    char label[48];
    double K;
    double T;
    bool is_call;
    double quantity;
    double initial_price;
};

struct PortfolioRiskResult {
    long long num_simulations;
    double portfolio_initial_value;
    double mean_simulated_loss;
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double max_loss;
    double max_gain;
    double kernel_time_ms;
    double total_time_ms;
};

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double black_scholes_price(double S0, double K, double r, double sigma, double T, bool is_call) {
    if (T <= 0.0) {
        return is_call ? std::max(S0 - K, 0.0) : std::max(K - S0, 0.0);
    }
    double sigma_sqrt_T = sigma * std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T;
    double d2 = d1 - sigma_sqrt_T;

    if (is_call) {
        return S0 * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    } else {
        return K * std::exp(-r * T) * normal_cdf(-d2) - S0 * normal_cdf(-d1);
    }
}
