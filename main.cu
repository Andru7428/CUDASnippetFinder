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

int main(int argc, char** argv) {
    bool self_join, computing_rows, computing_cols;
    size_t start_row = 0;
    size_t start_col = 0;
    
    int n = 7002;
    int m = 240;
    int l = 120;
    std::vector<double> Ta_h(n);

    std::ifstream is("WalkRun2_80_3800_240.txt");
    for (int i = 0; i < n; ++i)
    {
        is >> Ta_h[i];
    }


    int n_x = Ta_h.size() - l + 1;
    int n_y = n_x;

    if (n_x < 1 || n_y < 1) {
        printf("Error: window size must be smaller than the timeseries length\n");
        return 1;
    }
     
    SCAMP::SCAMPArgs args;
    args.window = l;
    args.has_b = false;
    args.profile_a.type = ParseProfileType("1NN_INDEX");
    args.profile_b.type = ParseProfileType("1NN_INDEX");
    args.precision_type = GetPrecisionType(false, true, false, false);
    args.profile_type = ParseProfileType("1NN_INDEX");
    args.timeseries_a = std::move(Ta_h);
    args.silent_mode = false;
    int size = pow(n - l + 1, 2) * sizeof(float);
    cudaMalloc(&args.distance_matrix, size);
    cudaMalloc(&args.MPdist_vector, (n - l + 1) * sizeof(float));
    /*
    try {
        InitProfileMemory(&args);
        SCAMP::do_SCAMP(&args);
    }
    catch (const SCAMPException& e) {
        std::cout << e.what() << "\n";
        exit(1);
    }

    float* distance_matrix = (float*)malloc(size);
    cudaMemcpy(distance_matrix, args.distance_matrix, size, cudaMemcpyDeviceToHost);

    float* h_mpdist = (float*)malloc((n - m) * sizeof(float));
    MPdist(args.distance_matrix, distance_matrix, h_mpdist, n, m, l);
    for (int i = 0; i < n - m; i++) {
        printf("%f ", sqrt(2 * l * (1 - h_mpdist[i])));
    }
    */
    int numThreads = 256;
    int numBlocks = n / 256 / 2 + 1;

    int N = (n + m - 1) / m - 1;
    float* h_profiles = (float*)malloc(N * (n - m) * sizeof(float));
    float* d_profiles;
    cudaMalloc(&d_profiles, N * (n - m) * sizeof(float));
    float* h_profile_area = (float*)malloc(N * (n - m) * sizeof(float));
    float* d_profile_area = (float*)malloc(numBlocks * sizeof(float));
    cudaMalloc(&d_profile_area, numBlocks * sizeof(float));

    std::ifstream is_2("profiles.txt");
    for (int i = 0; i < N * (n - m); i++)
    {
        is_2 >> h_profiles[i];
    }
    cudaMemcpy(d_profiles, h_profiles, N * (n - m) * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<Snippet> snippets = snippet_finder(d_profiles, n, m, 2);

    /*
    get_profile_area<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_profiles,  d_profiles + n - m, d_profile_area, n - m);
    cudaMemcpy(h_profile_area, d_profile_area, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < numBlocks; i++)
    {
        sum += h_profile_area[i];
    }
    printf("%f", sum);
    */
    /*
    float* distance_matrix = (float*)malloc(size);
    cudaMemcpy(distance_matrix, args.distance_matrix, size, cudaMemcpyDeviceToHost);
    std::ofstream os("matr_scamp.txt");
    for (int i = 0; i < pow(n - l + 1, 2); i++) {
        //printf("%f\n", sqrt(2 * m * (1 - distance_matrix[i])));
        //printf("%f\n", distance_matrix[i]);
        os << distance_matrix[i] << "\n";
    }
    os.close();

    WriteProfileToFile("profile", "index",
        args.profile_a, false, l,
        0, 0);
    */
    return 0;
}
