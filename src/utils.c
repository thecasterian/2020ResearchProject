#include "../include/utils.h"

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
 * @param len Length of array.
 * @param arr Array where to find the value.
 * @param val Value to find.
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

/**
 * @brief Returns ceil(num / den), e.g., divceil(7, 3) = ceil(7 / 3) = 3.
 *        \p num and \p den must be positive integers.
 *
 * @param num Numerator.
 * @param den Denominator.
 * @return Ceiling of quotient.
 */
int divceil(int num, int den) {
    return (num + den - 1) / den;
}
