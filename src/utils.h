#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "ibm3d.h"

/* Access to an element of an 1d coallocated array. */
#define c1e(a, i) ((a)[(i)+2])
/* Access to an element of an 3d coallocated grid. */
#define c3e(a, i, j, k) ((a)[(Ny+4)*(Nz+4)*((i)+2) + (Nz+4)*((j)+2) + (k)+2])
/* Access to an element of an 3d x-staggered grid. */
#define xse(a, i, j, k) ((a)[(Ny+4)*(Nz+4)*(i) + (Nz+4)*((j)+2) + (k)+2])
/* Access to an element of an 3d y-staggered grid. */
#define yse(a, i, j, k) ((a)[(Ny+1)*(Nz+4)*((i)+2) + (Nz+4)*(j) + (k)+2])
/* Access to an element of an 3d z-staggered grid. */
#define zse(a, i, j, k) ((a)[(Ny+4)*(Nz+1)*((i)+2) + (Nz+1)*((j)+2) + (k)])

#define FOR_INNER_CELL(i, j, k) \
    for (int i = 0; i < Nx; i++) \
        for (int j = 0; j < Ny; j++) \
            for (int k = 0; k < Nz; k++)
#define FOR_ALL_XSTAG(i, j, k) \
    for (int i = 0; i <= Nx; i++) \
        for (int j = 0; j < Ny; j++) \
            for (int k = 0; k < Nz; k++)
#define FOR_ALL_YSTAG(i, j, k) \
    for (int i = 0; i < Nx; i++) \
        for (int j = 0; j <= Ny; j++) \
            for (int k = 0; k < Nz; k++)
#define FOR_ALL_ZSTAG(i, j, k) \
    for (int i = 0; i < Nx; i++) \
        for (int j = 0; j < Ny; j++) \
            for (int k = 0; k <= Nz; k++)

/* Use for an unused variable or parameter to prevent a warning. */
#define UNUSED __attribute__((unused))

#define max(a, b) ({typeof(a) _a = a; typeof(b) _b = b; _a > _b ? _a : _b;})
#define min(a, b) ({typeof(a) _a = a; typeof(b) _b = b; _a < _b ? _a : _b;})

#define SWAP(a, b) do {typeof(a) tmp = a; a = b; b = tmp;} while (0)

FILE *fopen_check(const char *restrict filename, const char *restrict modes);

int lower_bound(const int, const double [], const double);
int upper_bound(const int, const double [], const double);

int dir_to_idx(IBMSolverDirection);

void IBMSolver_ghost_interp(
    IBMSolver *,
    const int, const int, const int,
    int [restrict][3], double [restrict]
);

int divceil(int, int);

#endif