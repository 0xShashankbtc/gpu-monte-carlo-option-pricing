#include "common.h"
#include <curand_kernel.h>
#include <chrono>

__device__ inline double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline void block_reduce_two_values(double val1, double val2, double* block_out1, double* block_out2) {
    static __shared__ double shared1[32];
    static __shared__ double shared2[32];

    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val1 = warp_reduce_sum(val1);
    val2 = warp_reduce_sum(val2);

    if (lane == 0) {
        shared1[wid] = val1;
        shared2[wid] = val2;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    double sum1 = (threadIdx.x < num_warps) ? shared1[lane] : 0.0;
    double sum2 = (threadIdx.x < num_warps) ? shared2[lane] : 0.0;

    if (wid == 0) {
        sum1 = warp_reduce_sum(sum1);
        sum2 = warp_reduce_sum(sum2);
        if (lane == 0) {
            *block_out1 = sum1;
            *block_out2 = sum2;
        }
    }
}

__global__ void monte_carlo_option_kernel(
    double S0,
    double K,
    double drift,
    double vol_term,
    bool is_call,
    long long N,
    unsigned long long seed,
    double* d_total_sum,
    double* d_total_sum_sq
) {
    unsigned long long tid = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, tid, 0, &rng_state);

    double thread_sum = 0.0;
    double thread_sum_sq = 0.0;

    for (long long i = tid; i < N; i += stride) {
        float z = curand_normal(&rng_state);
        double S_T = S0 * exp(drift + vol_term * (double)z);

        double payoff = 0.0;
        if (is_call) {
            payoff = (S_T > K) ? (S_T - K) : 0.0;
        } else {
            payoff = (K > S_T) ? (K - S_T) : 0.0;
        }

        thread_sum += payoff;
        thread_sum_sq += payoff * payoff;
    }

    __shared__ double block_sum;
    __shared__ double block_sum_sq;

    block_reduce_two_values(thread_sum, thread_sum_sq, &block_sum, &block_sum_sq);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(d_total_sum, block_sum);
        atomicAdd(d_total_sum_sq, block_sum_sq);
    }
}

MonteCarloResult run_gpu_option_mc(const OptionParams& params, long long num_simulations, unsigned long long seed = 123456789ULL) {
    MonteCarloResult result;
    result.num_simulations = num_simulations;

    double* d_total_sum = nullptr;
    double* d_total_sum_sq = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_total_sum, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_total_sum_sq, sizeof(double)));

    CUDA_CHECK(cudaMemset(d_total_sum, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_total_sum_sq, 0, sizeof(double)));

    double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
    double vol_term = params.sigma * std::sqrt(params.T);
    double discount = std::exp(-params.r * params.T);

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

    monte_carlo_option_kernel<<<num_blocks, threads_per_block>>>(
        params.S0,
        params.K,
        drift,
        vol_term,
        params.is_call,
        num_simulations,
        seed,
        d_total_sum,
        d_total_sum_sq
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    auto wall_stop = std::chrono::high_resolution_clock::now();

    float kernel_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event));
    result.kernel_time_ms = (double)kernel_time_ms;

    std::chrono::duration<double, std::milli> wall_dur = wall_stop - wall_start;
    result.total_time_ms = wall_dur.count();

    double h_total_sum = 0.0;
    double h_total_sum_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_total_sum, d_total_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_total_sum_sq, d_total_sum_sq, sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_total_sum));
    CUDA_CHECK(cudaFree(d_total_sum_sq));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    double mean_payoff = h_total_sum / (double)num_simulations;
    double var_payoff = 0.0;
    if (num_simulations > 1) {
        var_payoff = (h_total_sum_sq - (double)num_simulations * mean_payoff * mean_payoff) / (double)(num_simulations - 1);
        if (var_payoff < 0.0) var_payoff = 0.0;
    }

    result.estimated_price = discount * mean_payoff;
    result.standard_error = discount * std::sqrt(var_payoff / (double)num_simulations);

    result.ci95_lower = result.estimated_price - 1.959964 * result.standard_error;
    result.ci95_upper = result.estimated_price + 1.959964 * result.standard_error;

    result.ci99_lower = result.estimated_price - 2.575829 * result.standard_error;
    result.ci99_upper = result.estimated_price + 2.575829 * result.standard_error;

    result.bs_exact_price = black_scholes_price(params.S0, params.K, params.r, params.sigma, params.T, params.is_call);
    result.absolute_error = std::abs(result.estimated_price - result.bs_exact_price);
    result.relative_error_pct = (result.absolute_error / result.bs_exact_price) * 100.0;

    if (result.kernel_time_ms > 0.0) {
        result.throughput_sims_per_sec = ((double)num_simulations / (result.kernel_time_ms / 1000.0));
    } else {
        result.throughput_sims_per_sec = 0.0;
    }

    return result;
}
