#include "utils.h"

#include "mpi.h"

/**
 * @brief Opens the file. Abort MPI with exit status -1 if open failed.
 *
 * @param[in] filename Name of file.
 * @param[in] modes Access mode.
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
 * @param[in] len Length of array.
 * @param[in] arr Array where to find the value.
 * @param[in] val Value to find.
 *
 * @return Index found.
 *
 * @remark \p arr must not be NULL nor an array with length less than \p len.
 */
int lower_bound(const int len, const double arr[const static len], const double val) {
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
 * @param[in] len Length of array.
 * @param[in] arr Array where to find the value.
 * @param[in] val Value to find.
 *
 * @return Index found.
 *
 * @remark \p arr must not be NULL nor an array with length less than \p len.
 */
int upper_bound(const int len, const double arr[const static len], const double val) {
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
