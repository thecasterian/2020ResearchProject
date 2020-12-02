#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

#define FOR_ALL_CELL(i, j, k) \
    for (int i = 1; i <= Nx; i++) \
        for (int j = 1; j <= Ny; j++) \
            for (int k = 1; k <= Nz; k++)
#define FOR_ALL_XSTAG(i, j, k) \
    for (int i = 0; i <= Nx; i++) \
        for (int j = 1; j <= Ny; j++) \
            for (int k = 1; k <= Nz; k++)
#define FOR_ALL_YSTAG(i, j, k) \
    for (int i = 1; i <= Nx; i++) \
        for (int j = 0; j <= Ny; j++) \
            for (int k = 1; k <= Nz; k++)
#define FOR_ALL_ZSTAG(i, j, k) \
    for (int i = 1; i <= Nx; i++) \
        for (int j = 1; j <= Ny; j++) \
            for (int k = 0; k <= Nz; k++)

/* Convert local index to global index. */
#define LOCL_TO_GLOB(i) ((i) + solver->ilower - 1)
/* Convert global index to local index. */
#define GLOB_TO_LOCL(i) ((i) - solver->ilower + 1)

/* Global index of cell (i, j, k) where i is local index, starting from
   (ilower - 1) * Ny * Nz + 1. */
#define GLOB_CELL_IDX(i, j, k) (Ny*Nz*(LOCL_TO_GLOB(i)-1) + Nz*((j)-1) + (k))
/* Local index of cell (i, j, k) where i is local index, starting from 1. */
#define LOCL_CELL_IDX(i, j, k) (Ny*Nz*((i)-1) + Nz*((j)-1) + (k))

#define SWAP(a, b) do {typeof(a) tmp = a; a = b; b = tmp;} while (0)

FILE *fopen_check(const char *restrict filename, const char *restrict modes);

int lower_bound(const int, const double [], const double);
int upper_bound(const int, const double [], const double);

#endif