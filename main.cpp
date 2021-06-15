#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#endif

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

#ifdef _DISTRIBUTED_EXECUTION_
#include "../kubernetes/scamp_interface.h"
#endif

#ifdef _DISTRIBUTED_EXECUTION_
DEFINE_int64(distributed_tile_size, 4000000,
    "tile size to use for computation on worker notes");
DEFINE_string(hostname_port, "localhost:30078",
    "Hostname:Port of SCAMP server to perform distributed work");
#endif

int main(int argc, char** argv) {
    bool self_join, computing_rows, computing_cols;
    size_t start_row = 0;
    size_t start_col = 0;
    
    int n = 7000;
    int m = 240;
    std::vector<double> Ta_h(n);

    std::ifstream is("WalkRun2_80_3800_240.txt");
    for (int i = 0; i < n; ++i)
    {
        is >> Ta_h[i];
    }


    int n_x = Ta_h.size() - m + 1;
    int n_y = n_x;

    if (n_x < 1 || n_y < 1) {
        printf("Error: window size must be smaller than the timeseries length\n");
        return 1;
    }
     
    SCAMP::SCAMPArgs args;
    args.window = m;
    args.has_b = false;
    args.profile_a.type = ParseProfileType("1NN_INDEX");
    args.profile_b.type = ParseProfileType("1NN_INDEX");
    args.precision_type = GetPrecisionType(false, true, false, false);
    args.profile_type = ParseProfileType("1NN_INDEX");
    args.timeseries_a = std::move(Ta_h);
    args.silent_mode = false;
    int size = pow(n - m + 1, 2) * sizeof(float);
    cudaMalloc(&args.distance_matrix, size);

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
    std::ofstream os("matr_scamp.txt");
    for (int i = 0; i < pow(n - m + 1, 2); i++) {
        //printf("%f\n", sqrt(2 * m * (1 - distance_matrix[i])));
        //printf("%f\n", distance_matrix[i]);
        os << distance_matrix[i] << "\n";
    }
    os.close();

    WriteProfileToFile("profile", "index",
        args.profile_a, false, m,
        0, 0);

    return 0;
}
