#include <stdio.h>

#define Re 1000
#define dt 0.01

#define L 1
#define W 1

#define N1 16
#define N2 16
#define N3 16

typedef double mat[N1][N2][N3];
typedef double mat2[N1][N2+1][N3];

int main(void) {
    mat u1, u1_nxt, u1_hat;
    mat2 u2, u2_nxt, u2_hat;
    mat u3, u3_nxt, u3_hat;

    mat H1, H1_prv, H1_nxt;
    mat2 H2, H2_prv, H2_nxt;
    mat H3, H3_prv, H3_nxt;

    mat phi, phi_new;

    // solve for C

}