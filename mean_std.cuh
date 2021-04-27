/*
* mean_std.cuh
*
* Файл содержит интерфейсы следующих функций:
* mean_std - вычисление средних значений и среднеквадратических отклонений подпоследовательностей ряда
* 
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#ifndef SNIPPETFINDER_MEANSTD_H_
#define SNIPPETFINDER_MEANSTD_H_

template<class T>
__global__ void mean_std(T* g_cumsum, T* g_cumsum_sqr, float* g_mean, float* g_std, int l, int size);

#endif  // SNIPPETFINDER_MEANSTD_H_