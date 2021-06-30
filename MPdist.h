#pragma once

void computePba(float* d_distance_matrix, int* d_Pba, int n, int m, int l);

void computePab(float* d_distance_matrix, int* d_Pab, int n, int m, int l);

void MPdist(float* d_distance_matrix, float* h_distance_matrix, float* d_mpdist, int n, int m, int l);
