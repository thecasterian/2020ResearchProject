#ifndef UTILS_H
#define UTILS_H

#include "mpi.h"

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

/* Global index of cell (i, j, k) where i is local index, ranging from 1 to
   Nx_global * Ny * Nz. */
#define GLOB_CELL_IDX(i, j, k) (Ny*Nz*(LOCL_TO_GLOB(i)-1) + Nz*((j)-1) + (k))
/* Local index of cell (i, j, k) where i is local index, ranging from 1 to
   Nx * Ny * Nz. */
#define LOCL_CELL_IDX(i, j, k) (Ny*Nz*((i)-1) + Nz*((j)-1) + (k))

#define SWAP(a, b) do {typeof(a) tmp = a; a = b; b = tmp;} while (0)

static FILE *fopen_check(const char *restrict filename, const char *restrict modes) {
    FILE *fp = fopen(filename, modes);
    if (!fp) {
        fprintf(stderr, "error: cannot open file \"%s\"\n", filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return fp;
}

#endif