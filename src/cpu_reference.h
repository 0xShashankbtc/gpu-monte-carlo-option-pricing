#pragma once

#include "common.h"
#include <random>
#include <thread>
#include <vector>
#include <chrono>

inline MonteCarloResult run_cpu_option_mc(const OptionParams& params, long long num_simulations, int num_threads = 0) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 4;
    }

    MonteCarloResult result;
    result.num_simulations = num_simulations;

    double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
    double vol_term = params.sigma * std::sqrt(params.T);
    double discount = std::exp(-params.r * params.T);

    std::vector<double> thread_sums(num_threads, 0.0);
    std::vector<double> thread_sums_sq(num_threads, 0.0);
    std::vector<std::thread> workers;

    long long sims_per_thread = num_simulations / num_threads;
    long long rem = num_simulations % num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        long long count = sims_per_thread + (t < rem ? 1 : 0);
        unsigned int seed = 1337 + t * 9973;

        workers.emplace_back([&, t, count, seed]() {
            std::mt19937_64 rng(seed);
            std::normal_distribution<double> norm_dist(0.0, 1.0);

            double local_sum = 0.0;
            double local_sum_sq = 0.0;

            for (long long i = 0; i < count; ++i) {
                double z = norm_dist(rng);
                double S_T = params.S0 * std::exp(drift + vol_term * z);
                double payoff = 0.0;
                if (params.is_call) {
                    payoff = (S_T > params.K) ? (S_T - params.K) : 0.0;
                } else {
                    payoff = (params.K > S_T) ? (params.K - S_T) : 0.0;
                }
                local_sum += payoff;
                local_sum_sq += payoff * payoff;
            }

            thread_sums[t] = local_sum;
            thread_sums_sq[t] = local_sum_sq;
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    result.total_time_ms = duration.count();
    result.kernel_time_ms = duration.count();

    double total_sum = 0.0;
    double total_sum_sq = 0.0;
    for (int t = 0; t < num_threads; ++t) {
        total_sum += thread_sums[t];
        total_sum_sq += thread_sums_sq[t];
    }

    double mean_payoff = total_sum / (double)num_simulations;
    double var_payoff = 0.0;
    if (num_simulations > 1) {
        var_payoff = (total_sum_sq - (double)num_simulations * mean_payoff * mean_payoff) / (double)(num_simulations - 1);
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

    if (result.total_time_ms > 0.0) {
        result.throughput_sims_per_sec = ((double)num_simulations / (result.total_time_ms / 1000.0));
    } else {
        result.throughput_sims_per_sec = 0.0;
    }

    return result;
}
