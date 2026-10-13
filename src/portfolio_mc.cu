#include "common.h"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <chrono>

__device__ inline double device_norm_cdf(double x) {
    return normcdf(x);
}

__device__ inline double device_bs_price(double S, double K, double r, double sigma, double remaining_T, bool is_call) {
    if (remaining_T <= 1e-6) {
        return is_call ? fmax(S - K, 0.0) : fmax(K - S, 0.0);
    }
    double sigma_sqrt_T = sigma * sqrt(remaining_T);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * remaining_T) / sigma_sqrt_T;
    double d2 = d1 - sigma_sqrt_T;

    if (is_call) {
        return S * normcdf(d1) - K * exp(-r * remaining_T) * normcdf(d2);
    } else {
        return K * exp(-r * remaining_T) * normcdf(-d2) - S * normcdf(-d1);
    }
}

__global__ void portfolio_monte_carlo_kernel(
    const PortfolioAsset* __restrict__ d_assets,
    int num_assets,
    const PortfolioOptionPosition* __restrict__ d_positions,
    int num_positions,
    double risk_horizon_years,
    double portfolio_initial_value,
    long long N,
    unsigned long long seed,
    double* __restrict__ d_losses
) {
    unsigned long long tid = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, tid, 0, &rng_state);

    for (long long sim_idx = tid; sim_idx < N; sim_idx += stride) {
        double sim_asset_price[16];
        for (int a = 0; a < num_assets && a < 16; ++a) {
            float z = curand_normal(&rng_state);
            double drift = (d_assets[a].r - 0.5 * d_assets[a].sigma * d_assets[a].sigma) * risk_horizon_years;
            double vol_term = d_assets[a].sigma * sqrt(risk_horizon_years);
            sim_asset_price[a] = d_assets[a].S0 * exp(drift + vol_term * (double)z);
        }

        double sim_portfolio_value = 0.0;
        for (int p = 0; p < num_positions; ++p) {
            int a_idx = d_positions[p].asset_index;
            double S_horizon = sim_asset_price[a_idx];
            double rem_T = d_positions[p].T - risk_horizon_years;
            if (rem_T < 0.0) rem_T = 0.0;

            double opt_val = device_bs_price(
                S_horizon,
                d_positions[p].K,
                d_assets[a_idx].r,
                d_assets[a_idx].sigma,
                rem_T,
                d_positions[p].is_call
            );

            sim_portfolio_value += d_positions[p].quantity * opt_val;
        }

        double sim_loss = portfolio_initial_value - sim_portfolio_value;
        d_losses[sim_idx] = sim_loss;
    }
}

PortfolioRiskResult run_gpu_portfolio_risk(
    const std::vector<PortfolioAsset>& h_assets,
    const std::vector<PortfolioOptionPosition>& h_positions,
    double risk_horizon_years,
    long long num_simulations,
    unsigned long long seed = 987654321ULL,
    std::vector<double>* out_sample_losses = nullptr
) {
    PortfolioRiskResult result;
    result.num_simulations = num_simulations;

    int num_assets = (int)h_assets.size();
    int num_positions = (int)h_positions.size();

    double portfolio_init_val = 0.0;
    for (const auto& pos : h_positions) {
        double current_opt_price = black_scholes_price(
            h_assets[pos.asset_index].S0,
            pos.K,
            h_assets[pos.asset_index].r,
            h_assets[pos.asset_index].sigma,
            pos.T,
            pos.is_call
        );
        portfolio_init_val += pos.quantity * current_opt_price;
    }
    result.portfolio_initial_value = portfolio_init_val;

    PortfolioAsset* d_assets = nullptr;
    PortfolioOptionPosition* d_positions = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_assets, num_assets * sizeof(PortfolioAsset)));
    CUDA_CHECK(cudaMalloc((void**)&d_positions, num_positions * sizeof(PortfolioOptionPosition)));

    CUDA_CHECK(cudaMemcpy(d_assets, h_assets.data(), num_assets * sizeof(PortfolioAsset), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, h_positions.data(), num_positions * sizeof(PortfolioOptionPosition), cudaMemcpyHostToDevice));

    thrust::device_vector<double> d_losses(num_simulations);

    int threads_per_block = 256;
    int num_blocks = 1024;
    if (num_simulations < (long long)threads_per_block * num_blocks) {
        num_blocks = (int)((num_simulations + threads_per_block - 1) / threads_per_block);
        if (num_blocks < 1) num_blocks = 1;
    }

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    auto wall_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(start_event, 0));

    portfolio_monte_carlo_kernel<<<num_blocks, threads_per_block>>>(
        d_assets,
        num_assets,
        d_positions,
        num_positions,
        risk_horizon_years,
        portfolio_init_val,
        num_simulations,
        seed,
        thrust::raw_pointer_cast(d_losses.data())
    );
    CUDA_CHECK(cudaGetLastError());

    thrust::sort(d_losses.begin(), d_losses.end());

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    auto wall_stop = std::chrono::high_resolution_clock::now();

    float kernel_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event));
    result.kernel_time_ms = (double)kernel_time_ms;

    std::chrono::duration<double, std::milli> wall_dur = wall_stop - wall_start;
    result.total_time_ms = wall_dur.count();

    long long idx_95 = (long long)(0.95 * num_simulations);
    long long idx_99 = (long long)(0.99 * num_simulations);
    if (idx_95 >= num_simulations) idx_95 = num_simulations - 1;
    if (idx_99 >= num_simulations) idx_99 = num_simulations - 1;

    result.var_95 = d_losses[idx_95];
    result.var_99 = d_losses[idx_99];
    result.max_loss = d_losses[num_simulations - 1];
    result.max_gain = -d_losses[0];

    double tail_sum_95 = thrust::reduce(d_losses.begin() + idx_95, d_losses.end(), 0.0, thrust::plus<double>());
    double tail_sum_99 = thrust::reduce(d_losses.begin() + idx_99, d_losses.end(), 0.0, thrust::plus<double>());
    long long tail_count_95 = num_simulations - idx_95;
    long long tail_count_99 = num_simulations - idx_99;

    result.cvar_95 = (tail_count_95 > 0) ? (tail_sum_95 / tail_count_95) : result.var_95;
    result.cvar_99 = (tail_count_99 > 0) ? (tail_sum_99 / tail_count_99) : result.var_99;

    double total_loss_sum = thrust::reduce(d_losses.begin(), d_losses.end(), 0.0, thrust::plus<double>());
    result.mean_simulated_loss = total_loss_sum / (double)num_simulations;

    if (out_sample_losses != nullptr) {
        long long sample_size = std::min(num_simulations, 100000LL);
        out_sample_losses->resize(sample_size);
        long long step = num_simulations / sample_size;
        for (long long i = 0; i < sample_size; ++i) {
            (*out_sample_losses)[i] = d_losses[i * step];
        }
    }

    CUDA_CHECK(cudaFree(d_assets));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    return result;
}
