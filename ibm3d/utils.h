#ifndef UTILS_H
#define UTILS_H

#define FOR_ALL_CELL(i, j, k) \
    for (int i = 1; i <= Nx; i++) \
        for (int j = 1; j <= Ny; j++) \
            for (int k = 1; k <= Nz; k++)
#define FOR_ALL_XSTAG(i, j, k) \
    for (int i = 0; i <= Nx; i++) \
        for (int j = 0; j <= Ny+1; j++) \
            for (int k = 0; k <= Nz+1; k++)
#define FOR_ALL_YSTAG(i, j, k) \
    for (int i = 0; i <= Nx+1; i++) \
        for (int j = 0; j <= Ny; j++) \
            for (int k = 0; k <= Nz+1; k++)
#define FOR_ALL_ZSTAG(i, j, k) \
    for (int i = 0; i <= Nx+1; i++) \
        for (int j = 0; j <= Ny+1; j++) \
            for (int k = 0; k <= Nz; k++)

/* Index of cell (i, j, k), ranging from 1 to Nx * Ny * Nz. */
#define IDXFLAT(i, j, k) (Ny*Nz*((i)-1) + Nz*((j)-1) + (k))
#define SWAP(a, b) do {typeof(a) tmp = a; a = b; b = tmp;} while (0)

#endif