#include "utils.h"

#include "mpi.h"

/**
 * @brief Opens the file. Abort MPI with exit status -1 if open failed.
 *
 * @param filename Name of file.
 * @param modes Access mode.
 *
 * @return Pointer to the opened file.
 */
FILE *fopen_check(const char *restrict filename, const char *restrict modes) {
    FILE *fp = fopen(filename, modes);
    if (!fp) {
        fprintf(stderr, "error: cannot open file \"%s\"\n", filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return fp;
}

/**
 * @brief Finds the index of the first element in \p arr which is greater than
 *        or equal to \p val. \p arr must be sorted in increasing order.
 *
 * @param len Length of array.
 * @param arr Array where to find the value.
 * @param val Value to find.
 *
 * @return Index found.
 *
 * @remark \p arr must not be NULL nor an array with length less than \p len.
 */
int lower_bound_double(const int len, const double arr[const static len], const double val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val <= arr[mid]) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

/**
 * @brief Finds the index of the first element in \p arr which is greater than
 *        or equal to \p val. \p arr must be sorted in increasing order.
 *
 * @param len Length of array.
 * @param arr Array where to find the value.
 * @param val Value to find.
 *
 * @return Index found.
 *
 * @remark \p arr must not be NULL nor an array with length less than \p len.
 */
int lower_bound_int(const int len, const int arr[const static len], const int val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val <= arr[mid]) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

/**
 * @brief Finds the index of the first element in \p arr which is greater than
 *        \p val. \p arr must be sorted in increasing order.
 *
 * @param len Length of array.
 * @param arr Array where to find the value.
 * @param val Value to find.
 *
 * @return Index found.
 *
 * @remark \p arr must not be NULL nor an array with length less than \p len.
 */
int upper_bound_double(const int len, const double arr[const static len], const double val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val >= arr[mid]) {
            l = mid + 1;
        }
        else {
            h = mid;
        }
    }
    return l;
}

/**
 * @brief Finds the index of the first element in \p arr which is greater than
 *        \p val. \p arr must be sorted in increasing order.
 *
 * @param len Length of array.
 * @param arr Array where to find the value.
 * @param val Value to find.
 *
 * @return Index found.
 *
 * @remark \p arr must not be NULL nor an array with length less than \p len.
 */
int upper_bound_int(const int len, const int arr[const static len], const int val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val >= arr[mid]) {
            l = mid + 1;
        }
        else {
            h = mid;
        }
    }
    return l;
}

/**
 * @brief Converts IBMSolverDirection (2^0 - 2^5) to array index (0 - 5).
 *        Returns -1 if \p dir is an invalid value.
 *
 * @param dir IBMSolverDirection.
 * @return Corresponding index.
 */
int dir_to_idx(IBMSolverDirection dir) {
    switch (dir) {
    case DIR_NORTH:
        return 0;
    case DIR_EAST:
        return 1;
    case DIR_SOUTH:
        return 2;
    case DIR_WEST:
        return 3;
    case DIR_DOWN:
        return 4;
    case DIR_UP:
        return 5;
    default:;
    }
    return -1;
}

void IBMSolver_ghost_interp(
    IBMSolver *solver,
    const int i, const int j, const int k,
    int interp_idx[restrict][3], double interp_coeff[restrict]
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    /* Normal vector. */
    Vector n;
    /* Mirror point. */
    Vector m;

    n.x = (c3e(solver->lvset, i+1, j, k) - c3e(solver->lvset, i-1, j, k))
        / (c1e(solver->xc, i+1) - c1e(solver->xc, i-1));
    n.x = (c3e(solver->lvset, i, j+1, k) - c3e(solver->lvset, i, j-1, k))
        / (c1e(solver->yc, j+1) - c1e(solver->yc, j-1));
    n.x = (c3e(solver->lvset, i, j, k+1) - c3e(solver->lvset, i, j, k-1))
        / (c1e(solver->zc, k+1) - c1e(solver->zc, k-1));

    m = Vector_lincom(
        1, (Vector){c1e(solver->xc, i), c1e(solver->yc, j), c1e(solver->zc, k)},
        -2*c3e(solver->lvset, i, j, k), n
    );

    const int im = upper_bound_double(Nx+4, solver->xc, m.x) - 3;
    const int jm = upper_bound_double(Ny+4, solver->yc, m.y) - 3;
    const int km = upper_bound_double(Nz+4, solver->zc, m.z) - 3;

    /* Order of cells:
            011        111
             +----------+
        001 /|     101 /|          z
           +----------+ |          | y
           | |        | |          |/
           | +--------|-+          +------ x
           |/ 010     |/ 110
           +----------+
          000        100
    */
    for (int l = 0; l < 8; l++) {
        interp_idx[l][0] = im + !!(l & 4);
        interp_idx[l][1] = jm + !!(l & 2);
        interp_idx[l][2] = km + !!(l & 1);
    }

    const double xl = c1e(solver->xc, im), xu = c1e(solver->xc, im+1);
    const double yl = c1e(solver->yc, jm), yu = c1e(solver->yc, jm+1);
    const double zl = c1e(solver->zc, km), zu = c1e(solver->zc, km+1);
    const double vol = (xu - xl) * (yu - yl) * (zu - zl);

    interp_coeff[0] = (xu-m.x)*(yu-m.y)*(zu-m.z) / vol;
    interp_coeff[1] = (xu-m.x)*(yu-m.y)*(m.z-zl) / vol;
    interp_coeff[2] = (xu-m.x)*(m.y-yl)*(zu-m.z) / vol;
    interp_coeff[3] = (xu-m.x)*(m.y-yl)*(m.z-zl) / vol;
    interp_coeff[4] = (m.x-xl)*(yu-m.y)*(zu-m.z) / vol;
    interp_coeff[5] = (m.x-xl)*(yu-m.y)*(m.z-zl) / vol;
    interp_coeff[6] = (m.x-xl)*(m.y-yl)*(zu-m.z) / vol;
    interp_coeff[7] = (m.x-xl)*(m.y-yl)*(m.z-zl) / vol;
}
