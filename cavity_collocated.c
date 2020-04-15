#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Re 400
#define dt 0.005

#define N 64
#define numtstep 2500

typedef double mat[N+2][N+2];
typedef double mat1[N+1][N+2];
typedef double mat2[N+2][N+1];

mat p, p_next, p_prime;
mat u1, u1_next, u1_star, u1_tilde;
mat u2, u2_next, u2_star, u2_tilde;
mat1 U1, U1_next, U1_star;
mat2 U2, U2_next, U2_star;

mat1 N1, N1_prev;
mat2 N2, N2_prev;
mat Q;

double psi[N+1][N+1];

// for tdma
mat C1, C2, RHS1, RHS2;
double a[N+2], b[N+2], c[N+2], d[N+2], x[N+2];

const double h = 1. / N;
const double k = 1. + Re * h*h / dt;

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
    for (int i = 0; i <= N; i++) {
        U1[i][N+1] = 2;
    }
    for (int i = 1; i <= N; i++) {
        u1[i][N+1] = 2;
    }

    for (int tstep = 1; tstep <= numtstep; tstep++) {
        // calculate N
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                N1[i][j] = (U1[i][j] * (u1[i+1][j]+u1[i][j])/2 - U1[i-1][j] * (u1[i][j]+u1[i-1][j])/2) / h
                    + (U2[i][j] * (u1[i][j+1]+u1[i][j])/2 - U2[i][j-1] * (u1[i][j]+u1[i][j-1])/2) / h;
                N2[i][j] = (U1[i][j] * (u2[i+1][j]+u2[i][j])/2 - U1[i-1][j] * (u2[i][j]+u2[i-1][j])/2) / h
                    + (U2[i][j] * (u2[i][j+1]+u2[i][j])/2 - U2[i][j-1] * (u2[i][j]+u2[i][j-1])/2) / h;
            }
        }

        // calculate RHS
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                RHS1[i][j] = -dt/2 * (3*N1[i][j] - N1_prev[i][j])
                    - dt * (p[i+1][j]-p[i-1][j]) / (2*h)
                    + dt/Re * (u1[i+1][j]+u1[i-1][j]+u1[i][j+1]+u1[i][j-1]-4*u1[i][j]) / (h*h);
                RHS2[i][j] = -dt/2 * (3*N2[i][j] - N2_prev[i][j])
                    - dt * (p[i][j+1]-p[i][j-1]) / (2*h)
                    + dt/Re * (u2[i+1][j]+u2[i-1][j]+u2[i][j+1]+u2[i][j-1]-4*u2[i][j]) / (h*h);
            }
        }

        printf("%lf\n", dt/Re / (h*h));

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                printf("%lf ", RHS1[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                printf("%lf ", RHS2[i][j]);
            }
            printf("\n");
        }

        // calcuate C
        for (int j = 1; j <= N; j++) {
            fill(a+2, a+N+1, 1);
            b[1] = b[N] = -(2*k+1);
            fill(b+2, b+N, -2*k);
            fill(c+1, c+N, 1);
            for (int i = 1; i <= N; i++) {
                d[i] = -2*(k-1) * RHS1[i][j];
            }
            tdma(N);

            for (int i = 1; i <= N; i++) {
                C1[i][j] = x[i];
            }
        }
        for (int j = 1; j <= N; j++) {
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

        // calculate u_star
        for (int i = 1; i <= N; i++) {
            fill(a+2, a+N+1, 1);
            b[1] = b[N] = -(2*k+1);
            fill(b+2, b+N, -2*k);
            fill(c+1, c+N, 1);
            for (int j = 1; j <= N; j++) {
                d[j] = -2*(k-1) * C1[i][j];
            }
            tdma(N);

            for (int j = 1; j <= N; j++) {
                u1_star[i][j] = x[j] + u1[i][j];
            }
        }
        for (int i = 1; i <= N; i++) {
            fill(a+2, a+N+1, 1);
            b[1] = b[N] = -(2*k+1);
            fill(b+2, b+N, -2*k);
            fill(c+1, c+N, 1);
            for (int j = 1; j <= N; j++) {
                d[j] = -2*(k-1) * C2[i][j];
            }
            tdma(N);

            for (int j = 1; j <= N; j++) {
                u2_star[i][j] = x[j] + u2[i][j];
            }
        }

        // calculate u_tilde
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                u1_tilde[i][j] = u1_star[i][j] + dt * (p[i+1][j]-p[i-1][j]) / (2*h);
                u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1]-p[i][j-1]) / (2*h);
            }
        }

        // calculate U_star
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                U1_star[i][j] = (u1_tilde[i][j]+u1_tilde[i+1][j]) / 2
                    - dt * (p[i+1][j]-p[i][j]) / h;
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                U2_star[i][j] = (u2_tilde[i][j]+u2_tilde[i][j+1]) / 2
                    - dt * (p[i][j+1]-p[i][j]) / h;
            }
        }

        // calculate Q
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                Q[i][j] = (U1_star[i][j]-U1_star[i-1][j] + U2_star[i][j]-U2_star[i][j-1]) / (dt*h);
            }
        }

        // solve for p_prime
        memset(p_prime, 0, sizeof(p_prime));
        double res;
        for (int k = 1; k <= 1000000; k++) {
            res = 0;
            for (int i = 1; i <= N; i++) {
                p_prime[i][0] = p_prime[i][1];
                p_prime[i][N+1] = p_prime[i][N];
            }
            for (int j = 1; j <= N; j++) {
                p_prime[0][j] = p_prime[1][j];
                p_prime[N+1][j] = p_prime[N][j];
            }

            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (i == 1 && j == 1) {
                        continue;
                    }
                    double tmp = .25 * (p_prime[i+1][j]+p_prime[i][j+1]+p_prime[i-1][j]+p_prime[i][j-1] - h*h*Q[i][j]) - p_prime[i][j];
                    res += fabs(p_prime[i][j] - tmp);
                    p_prime[i][j] = tmp;
                }
            }
            if (res / N / N <= 1e-7) {
                // printf("%d\n", k);
                break;
            }
        }
        if (res / N / N > 1e-7) {
            printf("not converged!\n");
        }

        // calculate p_next
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                p_next[i][j] = p[i][j] + p_prime[i][j];
            }
        }

        // calculate u_next
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                u1_next[i][j] = u1_star[i][j] - dt * (p_prime[i+1][j]-p_prime[i-1][j]) / (2*h);
                u2_next[i][j] = u2_star[i][j] - dt * (p_prime[i][j+1]-p_prime[i][j-1]) / (2*h);
            }
        }

        // calculate U_next
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                U1_next[i][j] = U1_star[i][j] - dt * (p_prime[i+1][j]-p_prime[i][j]) / h;
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                U2_next[i][j] = U2_star[i][j] - dt * (p_prime[i][j+1]-p_prime[i][j]) / h;
            }
        }

        // velocity bc
        for (int i = 1; i <= N; i++) {
            u1_next[i][0] = -u1_next[i][1];
            u1_next[i][N+1] = 2 - u1_next[i][N];
            u2_next[i][0] = -u2_next[i][1];
            u2_next[i][N+1] = 2 - u2_next[i][N];
        }
        for (int j = 1; j <= N; j++) {
            u1_next[0][j] = -u1_next[1][j];
            u1_next[N+1][j] = -u1_next[N][j];
            u2_next[0][j] = -u2_next[1][j];
            u2_next[N+1][j] = -u2_next[N][j];
        }
        for (int i = 0; i <= N; i++) {
            U1_next[i][0] = -U1_next[i][1];
            U1_next[i][N+1] = 2 - U1_next[i][N];
        }
        for (int j = 0; j <= N; j++) {
            U2_next[0][j] = -U2_next[1][j];
            U2_next[N+1][j] = -U2_next[N][j];
        }

        // update for next time step
        memcpy(p, p_next, sizeof(p));
        memcpy(u1, u1_next, sizeof(u1));
        memcpy(u2, u2_next, sizeof(u2));
        memcpy(U1, U1_next, sizeof(U1));
        memcpy(U2, U2_next, sizeof(U2));
        memcpy(N1_prev, N1, sizeof(N1));
        memcpy(N2_prev, N2, sizeof(N2));

        if (tstep % 50 == 0) {
            printf("tstep: %d\n", tstep);
        }
    }

    // calculate streamfunction
    for (int i = 1; i <= N-1; i++) {
        for (int j = 1; j <= N-1; j++) {
            psi[i][j] = psi[i][j-1] + h * U1[i][j];
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