#include <stdio.h>

#define Re 100
#define dt 0.01

#define L 1
#define W 1

#define N1 16
#define N2 16

typedef double mat[N1][N2];
typedef double mat1[N1+1][N2];
typedef double mat2[N1][N2+1];

int main(void) {
    mat1 u1, u1_nxt, u1_hat;
    mat2 u2, u2_nxt, u2_hat;

    mat1 H1, H1_prv, H1_nxt;
    mat2 H2, H2_prv, H2_nxt;

    mat phi, phi_new;

    // solve for C

}