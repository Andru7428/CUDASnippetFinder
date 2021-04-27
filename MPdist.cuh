/*
* MPdist.cuh
*
* Файл содержит интерфейсы следующих функций:
* find_all_pab
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#ifndef SNIPPETFINDER_MPDIST_H_
#define SNIPPETFINDER_MPDIST_H_

__global__ void find_all_pab(float* g_ed_norm, float* g_all_pab, int m, int l, int n);

#endif  // SNIPPETFINDER_MPDIST_H_