#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Re 400
#define dt 0.005

#define N 32
#define beta 1
#define numtstep 10

double xc[N+2], yc[N+2];
double dx[N+2], dy[N+2];
double dx_bar[N+2], dy_bar[N+2];

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
double kx[N+2], ky[N+2];
double a[N+2], b[N+2], c[N+2], d[N+2], x[N+2];

double tanh_stretch(double xi) {
    return 1 - tanh(beta * (1 - 2*xi)) / tanh(beta);
}

double sq(double x) {
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
    // generate grid
    for (int i = 1; i <= N; i++) {
        // double left = tanh_stretch(1.* (i-1) / N);
        // double right = tanh_stretch(1. * i / N);

        double left = 1. * (i-1) / N;
        double right = 1. * i / N;

        dx[i] = dy[i] = right - left;
        xc[i] = yc[i] = (right + left) / 2;
    }

    // grid boundaries
    dx[0] = dx[1]; dx[N+1] = dx[N];
    xc[0] = -xc[1]; xc[N+1] = 2 - xc[N];
    dy[0] = dy[1]; dy[N+1] = dy[N];
    yc[0] = -yc[1]; yc[N+1] = 2 - yc[N];
    for (int i = 1; i <= N; i++) {
        dx_bar[i] = (xc[i+1]-xc[i-1]) / 2;
        dy_bar[i] = (yc[i+1]-yc[i-1]) / 2;
    }

    // tdma coefficients
    for (int i = 1; i <= N; i++) {
        kx[i] = 1 + Re * sq(dx_bar[i]) / dt;
        ky[i] = 1 + Re * sq(dy_bar[i]) / dt;
    }

    // initialize flow
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
                N1[i][j] = (U1[i][j] * (u1[i+1][j]*dx[i]+u1[i][j]*dx[i+1])/(dx[i]+dx[i+1])
                            - U1[i-1][j] * (u1[i][j]*dx[i-1]+u1[i-1][j]*dx[i])/(dx[i-1]+dx[i])) / dx[i]
                    + (U2[i][j] * (u1[i][j+1]*dy[j]+u1[i][j]*dy[j+1])/(dy[j]+dy[j+1])
                       - U2[i][j-1] * (u1[i][j]*dy[j-1]+u1[i][j-1]*dy[j])/(dy[j-1]+dy[j])) / dy[j];
                N2[i][j] = (U1[i][j] * (u2[i+1][j]*dx[i]+u2[i][j]*dx[i+1])/(dx[i]+dx[i+1])
                            - U1[i-1][j] * (u2[i][j]*dx[i-1]+u2[i-1][j]*dx[i])/(dx[i-1]+dx[i])) / dx[i]
                    + (U2[i][j] * (u2[i][j+1]*dy[j]+u2[i][j]*dy[j+1])/(dy[j]+dy[j+1])
                       - U2[i][j-1] * (u2[i][j]*dy[j-1]+u2[i][j-1]*dy[j])/(dy[j-1]+dy[j])) / dy[j];
            }
        }

        // calculate RHS
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                RHS1[i][j] = -dt/2 * (3*N1[i][j] - N1_prev[i][j])
                    - dt * (p[i+1][j]-p[i-1][j]) / (2*dx_bar[i])
                    + dt/Re * ((u1[i-1][j]-2*u1[i][j]+u1[i+1][j]) / sq(dx_bar[i])
                               + (u1[i][j-1]-2*u1[i][j]+u1[i][j+1]) / sq(dy_bar[j]));
                RHS2[i][j] = -dt/2 * (3*N2[i][j] - N2_prev[i][j])
                    - dt * (p[i][j+1]-p[i][j-1]) / (2*dy_bar[j])
                    + dt/Re * ((u2[i-1][j]-2*u2[i][j]+u2[i+1][j]) / sq(dx_bar[i])
                                 + (u2[i][j-1]-2*u2[i][j]+u2[i][j+1]) / sq(dy_bar[j]));
            }
        }

        // calcuate C
        for (int j = 1; j <= N; j++) {
            fill(a+2, a+N+1, 1);
            b[1] = -(2*kx[1]+1);
            for (int i = 2; i <= N-1; i++) {
                b[i] = -2*kx[i];
            }
            b[N] = -(2*kx[N]+1);
            fill(c+1, c+N, 1);
            for (int i = 1; i <= N; i++) {
                d[i] = -2*(kx[i]-1) * RHS1[i][j];
            }
            tdma(N);

            for (int i = 1; i <= N; i++) {
                C1[i][j] = x[i];
            }
        }
        for (int j = 1; j <= N; j++) {
            fill(a+2, a+N+1, 1);
            b[1] = -(2*kx[1]+1);
            for (int i = 2; i <= N-1; i++) {
                b[i] = -2*kx[i];
            }
            b[N] = -(2*kx[N]+1);
            fill(c+1, c+N, 1);
            for (int i = 1; i <= N; i++) {
                d[i] = -2*(kx[i]-1) * RHS2[i][j];
            }
            tdma(N);

            for (int i = 1; i <= N; i++) {
                C2[i][j] = x[i];
            }
        }

        // calculate u_star
        for (int i = 1; i <= N; i++) {
            fill(a+2, a+N+1, 1);
            b[1] = -(2*ky[1]+1);
            for (int j = 2; j <= N-1; j++) {
                b[j] = -2*ky[j];
            }
            b[N] = -(2*ky[N]+1);
            fill(c+1, c+N, 1);
            for (int j = 1; j <= N; j++) {
                d[j] = -2*(ky[j]-1) * C1[i][j];
            }
            tdma(N);

            for (int j = 1; j <= N; j++) {
                u1_star[i][j] = x[j] + u1[i][j];
            }
        }
        for (int i = 1; i <= N; i++) {
            fill(a+2, a+N+1, 1);
            b[1] = -(2*ky[1]+1);
            for (int j = 2; j <= N-1; j++) {
                b[j] = -2*ky[j];
            }
            b[N] = -(2*ky[N]+1);
            fill(c+1, c+N, 1);
            for (int j = 1; j <= N; j++) {
                d[j] = -2*(ky[j]-1) * C2[i][j];
            }
            tdma(N);

            for (int j = 1; j <= N; j++) {
                u2_star[i][j] = x[j] + u2[i][j];
            }
        }

        // calculate u_tilde
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                u1_tilde[i][j] = u1_star[i][j] + dt * (p[i+1][j]-p[i-1][j]) / (2*dx_bar[i]);
                u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1]-p[i][j-1]) / (2*dy_bar[j]);
            }
        }

        // calculate U_star
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                U1_star[i][j] = (dx[i+1]*u1_tilde[i][j]+dx[i]*u1_tilde[i+1][j]) / (dx[i]+dx[i+1]);
                    - dt * (p[i+1][j]-p[i][j]) / (xc[i+1]-xc[i]);
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                U2_star[i][j] = (dy[j+1]*u2_tilde[i][j]+dy[j]*u2_tilde[i][j+1]) / (dy[j]+dy[j+1]);
                    - dt * (p[i][j+1]-p[i][j]) / (yc[j+1]-yc[j]);
            }
        }

        // calculate Q
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                Q[i][j] = (U1_star[i][j]-U1_star[i-1][j]) / (dt*dx[i])
                    + (U2_star[i][j]-U2_star[i][j-1]) / (dt*dy[j]);
            }
        }

        // solve for p_prime
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
                    p_prime[i][j]
                        = ((p_prime[i-1][j]+p_prime[i+1][j])/sq(dx_bar[i])
                           + (p_prime[i][j-1]+p_prime[i][j+1])/sq(dy_bar[j]) - Q[i][j])
                        / (2/sq(dx_bar[i]) + 2/sq(dy_bar[j]));
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
                u1_next[i][j] = u1_star[i][j] - dt * (p_prime[i+1][j]-p_prime[i-1][j]) / (2*dx_bar[i]);
                u2_next[i][j] = u2_star[i][j] - dt * (p_prime[i][j+1]-p_prime[i][j-1]) / (2*dy_bar[j]);
            }
        }

        // calculate U_next
        for (int i = 1; i <= N-1; i++) {
            for (int j = 1; j <= N; j++) {
                U1_next[i][j] = U1_star[i][j] - dt * (p_prime[i+1][j]-p_prime[i][j]) / (xc[i+1]-xc[i]);
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N-1; j++) {
                U2_next[i][j] = U2_star[i][j] - dt * (p_prime[i][j+1]-p_prime[i][j]) / (yc[j+1]-yc[j]);
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
            psi[i][j] = psi[i][j-1] + dy[j] * U1[i][j];
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