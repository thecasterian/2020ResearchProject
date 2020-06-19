#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Re 400
#define dt 0.01

#define N 128
#define numtstep 1000

typedef double mat[N+2][N+2];
typedef double mat1[N+1][N+2];
typedef double mat2[N+2][N+1];

mat1 u1, u1_nxt, u1_hat;
mat2 u2, u2_nxt, u2_hat;

mat1 H1, H1_prv;
mat2 H2, H2_prv;

mat phi_nxt;

double psi[N+1][N+1];

// for tdma
mat1 C1, RHS1;
mat2 C2, RHS2;
double a[N+2], b[N+2], c[N+2], d[N+2], x[N+2];

// for poisson
mat Q;

const double h = 1. / N;
const double k = 1 + Re*h*h/dt;

inline double sq(double x) {
    return x * x;
}

double fill(double *begin, double *end, double value) {
    for (double *p = begin; p != end; p++)
        *p = value;
}

double tdma(int n) {
    for (int k = 2; k <= n; k++) {
        double m = a[k] / b[k-1];
        b[k] -= m * c[k-1];
        d[k] -= m * d[k-1];
    }
    x[n] = d[n] / b[n];
    for (int k = n-1; k >= 1; k--) {
        x[k] = (d[k] - c[k] * x[k+1]) / b[k];
    }
}

int main(void) {
    // initialize
    for (int i = 1; i <= N-1; i++) {
        u1[i][N+1] = 2;
    }

    for (int tstep = 1; tstep <= numtstep; tstep++) {
        // calculate H
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                H1[i][j] = -(sq(u1[i+1][j]) - sq(u1[i-1][j])) / (2*h)
                    -((u1[i][j+1]+u1[i][j])*(u2[i][j]+u2[i+1][j]) - (u1[i][j]+u1[i][j-1])*(u2[i][j-1]+u2[i+1][j-1])) / (4*h);
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                H2[i][j] = -((u2[i][j]+u2[i+1][j])*(u1[i][j+1]+u1[i][j]) - (u2[i][j]+u2[i-1][j])*(u1[i-1][j+1]+u1[i-1][j])) / (4*h)
                    -(sq(u2[i][j+1]) - sq(u2[i][j-1])) / (2*h);
            }
        }

        // calculate RHS
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                RHS1[i][j] = dt/2 * (3*H1[i][j] - H1_prv[i][j])
                    + dt/Re * ((u1[i-1][j]-2*u1[i][j]+u1[i+1][j])/sq(h) + (u1[i][j-1]-2*u1[i][j]+u1[i][j+1])/sq(h));
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                RHS2[i][j] = dt/2 * (3*H2[i][j] - H2_prv[i][j])
                    + dt/Re * ((u2[i-1][j]-2*u2[i][j]+u2[i+1][j])/sq(h) + (u2[i][j-1]-2*u2[i][j]+u2[i][j+1])/sq(h));
            }
        }

        // solve for C
        for (int j = 1; j <= N; j++) {
            fill(a+2, a+N, 1);
            fill(b+1, b+N, -2*k);
            fill(c+1, c+N-1, 1);
            for (int i = 1; i <= N-1; i++) {
                d[i] = -2*(k-1) * RHS1[i][j];
            }
            tdma(N-1);
            for (int i = 1; i <= N-1; i++) {
                C1[i][j] = x[i];
            }
        }
        for (int j = 1; j <= N-1; j++) {
            fill(a+2, a+N+1, 1);
            b[1] = b[N] = -(2*k+1);
            fill(b+2, b+N, -2*k);
            fill(c+1, c+N, 1);
            for (int i = 1; i <= N; i++) {
                d[i] = -2*(k-1) * RHS2[i][j];
            }
            tdma(N);
            for (int i = 1; i <= N; i++) {
                C2[i][j] = x[i];
            }
        }

        // solve for u_hat
        for (int i = 1; i <= N-1; i++) {
            fill(a+2, a+N+1, 1);
            b[1] = b[N] = -(2*k+1);
            fill(b+2, b+N, -2*k);
            fill(c+1, c+N, 1);
            for (int j = 1; j <= N; j++) {
                d[j] = -2*(k-1) * C1[i][j];
            }
            tdma(N);
            for (int j = 1; j <= N; j++) {
                u1_hat[i][j] = x[j] + u1[i][j];
            }
        }
        for (int i = 1; i <= N; i++) {
            fill(a+2, a+N, 1);
            fill(b+1, b+N, -2*k);
            fill(c+1, c+N-1, 1);
            for (int j = 1; j <= N-1; j++) {
                d[j] = -2*(k-1) * C2[i][j];
            }
            tdma(N-1);
            for (int j = 1; j <= N-1; j++) {
                u2_hat[i][j] = x[j] + u2[i][j];
            }
        }
        for (int i = 1; i <= N-1; i++) {
            u1_hat[i][0] = -u1_hat[i][1];
            u1_hat[i][N+1] = 2 - u1_hat[i][N];
        }
        for (int j = 1; j <= N-1; j++) {
            u2_hat[0][j] = -u2_hat[1][j];
            u2_hat[N+1][j] = -u2_hat[N][j];
        }

        // calculate Q
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                Q[i][j] = 1/dt * ((u1_hat[i][j] - u1_hat[i-1][j])/h + (u2_hat[i][j] - u2_hat[i][j-1])/h);
            }
        }

        // solve for phi_nxt
        double res;
        for (int k = 1; k <= 300; k++) {
            res = 0;
            for (int i = 1; i <= N; i++) {
                phi_nxt[i][0] = phi_nxt[i][1];
                phi_nxt[i][N+1] = phi_nxt[i][N];
            }
            for (int j = 1; j <= N; j++) {
                phi_nxt[0][j] = phi_nxt[1][j];
                phi_nxt[N+1][j] = phi_nxt[N][j];
            }
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (i == 1 && j == 1) {
                        continue;
                    }
                    double tmp = .25 * (phi_nxt[i+1][j] + phi_nxt[i][j+1] + phi_nxt[i-1][j] + phi_nxt[i][j-1] - h*h*Q[i][j]);
                    res += fabs(phi_nxt[i][j] - tmp);
                    phi_nxt[i][j] = tmp;
                }
            }
            if (res / N / N <= 1e-7) {
                break;
            }
        }

        // solve for u_nxt
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                u1_nxt[i][j] = u1_hat[i][j] - dt * (phi_nxt[i+1][j] - phi_nxt[i][j]) / h;
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                u2_nxt[i][j] = u2_hat[i][j] - dt * (phi_nxt[i][j+1] - phi_nxt[i][j]) / h;
            }
        }
        for (int i = 1; i <= N-1; i++) {
            u1_nxt[i][0] = -u1_nxt[i][1];
            u1_nxt[i][N+1] = 2 - u1_nxt[i][N];
        }
        for (int j = 1; j <= N-1; j++) {
            u2_nxt[0][j] = -u2_nxt[1][j];
            u2_nxt[N+1][j] = -u2_nxt[N][j];
        }

        // update for next time step
        memcpy(u1, u1_nxt, sizeof(u1));
        memcpy(u2, u2_nxt, sizeof(u2));
        memcpy(H1_prv, H1, sizeof(H1));
        memcpy(H2_prv, H2, sizeof(H2));
    }

    // calculate psi (streamfunction)
    for (int i = 1; i <= N-1; i++) {
        for (int j = 1; j <= N-1; j++) {
            psi[i][j] = psi[i][j-1] + h * u1[i][j];
        }
    }

    FILE *fp = fopen("cavity_result.txt", "w");
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j <= N; j++) {
            fprintf(fp, "%.14lf ", psi[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}