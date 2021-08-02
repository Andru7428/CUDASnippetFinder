#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "SCAMP.h"
#include "common.h"
#include "scamp_exception.h"
#include "scamp_utils.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include "MPdist.h"
#include "SnippetFinder.h"
#include <chrono>

int main(int argc, char** argv) {
    bool self_join, computing_rows, computing_cols;
    size_t start_row = 0;
    size_t start_col = 0;
    
    int n = 20000;
    int m = 900;
    int l = m / 2;
    
    std::vector<double> Ta_h(n);

    std::ifstream is("WildVTrainedBird_10001_900.txt");
    for (int i = 0; i < n; i++)
    {
        is >> Ta_h[i];
    }

    auto start = std::chrono::high_resolution_clock::now();

    int n_x = Ta_h.size() - l + 1;
    int n_y = n_x;

    if (n_x < 1 || n_y < 1) {
        printf("Error: window size must be smaller than the timeseries length\n");
        return 1;
    }
    int N = (n + m - 1) / m - 1;
    unsigned __int64 size = (n - l + 1) * (m - l + 1) * sizeof(float);
    float* d_profiles;
    cudaMalloc(&d_profiles, N * (n - m) * sizeof(float));
    size = pow(n - l + 1, 2) * sizeof(float);
    SCAMP::SCAMPArgs args;
    args.window = l;
    args.has_b = false;
    args.profile_a.type = ParseProfileType("1NN_INDEX");
    args.profile_b.type = ParseProfileType("1NN_INDEX");
    args.precision_type = GetPrecisionType(false, true, false, false);
    args.profile_type = ParseProfileType("1NN_INDEX");
    args.timeseries_a = Ta_h;
    args.silent_mode = false;
    cudaError_t code = cudaMalloc(&args.distance_matrix, size);
    if (code != cudaSuccess)
    {
        printf("Memory error");
    }
    try {
        InitProfileMemory(&args);
        SCAMP::do_SCAMP(&args);
    }
    catch (const SCAMPException& e) {
        std::cout << e.what() << "\n";
        exit(1);
    }
    /*
    for (int i = 0; i < N; i++) {
        auto first = Ta_h.cbegin() + i * m;
        auto last = Ta_h.cbegin() + (i * m + m);
        std::vector<double> Tb_h(first, last);
        args.timeseries_b = std::move(Tb_h);
        try {
            InitProfileMemory(&args);
            SCAMP::do_SCAMP(&args);
        }
        catch (const SCAMPException& e) {
            std::cout << e.what() << "\n";
            exit(1);
        }

        MPdist_(args.distance_matrix, d_profiles, n, m, l, i);
    }
   
    //unsigned __int64 size = pow(n - l + 1, 2) * sizeof(float);

    //float* distance_matrix = (float*)malloc(size);
    //cudaMemcpy(distance_matrix, args.distance_matrix, size, cudaMemcpyDeviceToHost);
    */
    /*
    
    float* h_profiles = (float*)malloc(N * (n - m) * sizeof(float));
    MPdist(args.distance_matrix, distance_matrix, h_profiles, n, m, l);
    float* d_profiles;
    cudaMalloc(&d_profiles, N * (n - m) * sizeof(float));
    cudaMemcpy(d_profiles, h_profiles, N * (n - m) * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<Snippet> snippets = snippet_finder(d_profiles, n, m, 2);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    printf("%d", duration.count());
    */

    //for (Snippet& it : snippets) {
    //    printf("idx: %d, frac: %f\n", it.index, it.frac);
    //}
    
    return 0;
}
