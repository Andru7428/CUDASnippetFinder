/*
* ed_norm.cuh
* 
* Файл содержит интерфейсы следующих функций:
* ed_norm - вычисление матрицы евклидовых расстояний между подпоследовательностями временного ряда
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#ifndef SNIPPETFINDER_EDNORM_H_
#define SNIPPETFINDER_EDNORM_H_

template<class T>
__global__ void ed_norm(T* g_idata, float* g_mean, float* g_std, float* g_ed_norm, int m, int size);

#endif  // SNIPPETFINDER_EDNORM_H_