#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define RESET_ATTR      "\x1B[0m"
#define BOLD_TEXT       "\x1B[1m"
#define RED_TEXT        "\x1B[31m"
#define GREEN_TEXT      "\x1B[32m"
#define YELLOW_TEXT     "\x1B[33m"
#define BLUE_TEXT       "\x1B[34m"
#define MAGENTA_TEXT    "\x1B[35m"
#define CYAN_TEXT       "\x1B[36m"
#define WHITE_TEXT      "\x1B[37m"

void tdma(int n, double *a, double *b, double *c, double *d, double *e, double *x, double *y) {
    for (int k = 2; k <= n; k++) {
        double m = a[k] / b[k-1];
        b[k] -= m * c[k-1];
        d[k] -= m * d[k-1];
        e[k] -= m * e[k-1];
    }
    x[n] = d[n] / b[n];
    y[n] = e[n] / b[n];
    for (int k = n-1; k >= 1; k--) {
        x[k] = (d[k] - c[k] * x[k+1]) / b[k];
        y[k] = (e[k] - c[k] * y[k+1]) / b[k];
    }
}

int main(void) {
    /* Read input file */
    int Nx, Ny;
    double Re, dt;
    int numtstep;

    FILE *fp_in = fopen("cavity.in", "r");

    assert(fscanf(fp_in, "%*s %d", &Nx) == 1);
    double xf[Nx+1];
    for (int i = 0; i <= Nx; i++) {
        assert(fscanf(fp_in, "%lf", &xf[i]) == 1);
    }

    assert(fscanf(fp_in, "%*s %d", &Ny) == 1);
    double yf[Ny+1];
    for (int j = 0; j <= Ny; j++) {
        assert(fscanf(fp_in, "%lf", &yf[j]) == 1);
    }

    assert(fscanf(fp_in, "%*s %lf", &Re) == 1);
    assert(fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep) == 2);

    fclose(fp_in);

    /* Define variables */
    const int Nm = Nx > Ny ? Nx : Ny;

    double xc[Nx+2], dx[Nx+2];
    double yc[Ny+2], dy[Ny+2];

    double p[Nx+2][Ny+2], p_next[Nx+2][Ny+2], p_prime[Nx+2][Ny+2];
    double u1[Nx+2][Ny+2], u1_next[Nx+2][Ny+2], u1_star[Nx+2][Ny+2], u1_tilde[Nx+2][Ny+2];
    double u2[Nx+2][Ny+2], u2_next[Nx+2][Ny+2], u2_star[Nx+2][Ny+2], u2_tilde[Nx+2][Ny+2];

    double U1[Nx+1][Ny+2], U1_next[Nx+1][Ny+2], U1_star[Nx+1][Ny+2];
    double U2[Nx+2][Ny+1], U2_next[Nx+2][Ny+1], U2_star[Nx+2][Ny+1];

    double N1[Nx+2][Ny+2], N1_prev[Nx+2][Ny+2];
    double N2[Nx+2][Ny+2], N2_prev[Nx+2][Ny+2];
    double Q[Nx+2][Ny+2];
    double psi[Nx+1][Ny+1];

    double C1[Nx+2][Ny+2], C2[Nx+2][Ny+2], RHS1[Nx+2][Ny+2], RHS2[Nx+2][Ny+2];
    double kx_W[Nx+2], kx_P[Nx+2], kx_E[Nx+2];
    double ky_S[Ny+2], ky_P[Ny+2], ky_N[Ny+2];
    double a[Nm+2], b[Nm+2], c[Nm+2], d[Nm+2], e[Nm+2], x[Nm+2], y[Nm+2];

    /* Initialize */
    for (int i = 0; i < Nx+2; i++) {
        for (int j = 0; j < Ny+2; j++) {
            p[i][j] = p_next[i][j] = p_prime[i][j] = 0;
            u1[i][j] = u1_next[i][j] = u1_star[i][j] = u1_tilde[i][j] = 0;
            u2[i][j] = u2_next[i][j] = u2_star[i][j] = u2_tilde[i][j] = 0;
        }
    }
    for (int i = 0; i < Nx+1; i++) {
        for (int j = 0; j < Ny+2; j++) {
            U1[i][j] = U1_next[i][j] = U1_star[i][j] = 0;
        }
    }
    for (int i = 0; i < Nx+2; i++) {
        for (int j = 0; j < Ny+1; j++) {
            U2[i][j] = U2_next[i][j] = U2_star[i][j] = 0;
        }
    }

    /* Calculate grid variables */
    for (int i = 1; i <= Nx; i++) {
        dx[i] = xf[i] - xf[i-1];
        xc[i] = (xf[i] + xf[i-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        dy[j] = yf[j] - yf[j-1];
        yc[j] = (yf[j] + yf[j-1]) / 2;
    }
    dx[0] = dx[1];
    dx[Nx+1] = dx[Nx];
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];
    xc[0] = 2*xf[0] - xc[1];
    xc[Nx+1] = 2*xf[Nx] - xc[Nx];
    yc[0] = 2*yf[0] - yc[1];
    yc[Ny+1] = 2*yf[Ny] - yc[Ny];

    /* Calculate tdma coefficients */
    for (int i = 1; i <= Nx; i++) {
        kx_W[i] = dt / (2*Re * (xc[i] - xc[i-1])*dx[i]);
        kx_E[i] = dt / (2*Re * (xc[i+1] - xc[i])*dx[i]);
        kx_P[i] = kx_W[i] + kx_E[i] + 1;
    }
    for (int j = 1; j <= Ny; j++) {
        ky_S[j] = dt / (2*Re * (yc[j] - yc[j-1])*dy[j]);
        ky_N[j] = dt / (2*Re * (yc[j+1] - yc[j])*dy[j]);
        ky_P[j] = ky_S[j] + ky_N[j] + 1;
    }

    // initialize flow
    for (int i = 0; i <= Nx; i++) {
        U1[i][Ny+1] = 2;
    }
    for (int i = 1; i <= Nx; i++) {
        u1[i][Ny+1] = 2;
    }
    memcpy(N1_prev, N1, sizeof(double)*(Nx+2)*(Ny+2));
    memcpy(N2_prev, N2, sizeof(double)*(Nx+2)*(Ny+2));

    for (int tstep = 1; tstep <= numtstep; tstep++) {
        // calculate N
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u1_e = (u1[i][j] * dx[i+1] + u1[(i+1)][j] * dx[i]) / (dx[i] + dx[i+1]);
                double u2_e = (u2[i][j] * dx[i+1] + u2[(i+1)][j] * dx[i]) / (dx[i] + dx[i+1]);
                double u1_w = (u1[(i-1)][j] * dx[i] + u1[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
                double u2_w = (u2[(i-1)][j] * dx[i] + u2[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
                double u1_n = (u1[i][j] * dy[j+1] + u1[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
                double u2_n = (u2[i][j] * dy[j+1] + u2[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
                double u1_s = (u1[i][j-1] * dy[j] + u1[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);
                double u2_s = (u2[i][j-1] * dy[j] + u2[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);

                N1[i][j] = (U1[i][j]*u1_e - U1[(i-1)][j]*u1_w) / dx[i]
                    + (U2[i][j]*u1_n - U2[i][j-1]*u1_s) / dy[j];
                N2[i][j] = (U1[i][j]*u2_e - U1[(i-1)][j]*u2_w) / dx[i]
                    + (U2[i][j]*u2_n - U2[i][j-1]*u2_s) / dy[j];
            }
        }

        // calculate RHS
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                RHS1[i][j] = -dt/2 * (3*N1[i][j] - N1_prev[i][j])
                    - dt * (p[(i+1)][j] - p[(i-1)][j]) / (xc[i+1] - xc[i-1])
                    + 2 * (kx_W[i]*u1[(i-1)][j] + kx_E[i]*u1[(i+1)][j]
                           + ky_S[j]*u1[i][j-1] + ky_N[j]*u1[i][j+1]
                           - (kx_W[i]+kx_E[i]+ky_S[j]+ky_N[j])*u1[i][j]);
                RHS2[i][j] = -dt/2 * (3*N2[i][j] - N2_prev[i][j])
                    - dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1])
                    + 2 * (kx_W[i]*u2[(i-1)][j] + kx_E[i]*u2[(i+1)][j]
                           + ky_S[j]*u2[i][j-1] + ky_N[j]*u2[i][j+1]
                           - (kx_W[i]+kx_E[i]+ky_S[j]+ky_N[j])*u2[i][j]);
            }
        }

        // calcuate C
        for (int j = 1; j <= Ny; j++) {
            for (int i = 2; i <= Nx; i++) {
                a[i] = -kx_W[i];
            }
            b[1] = kx_W[1] + kx_P[1];
            for (int i = 2; i <= Nx-1; i++) {
                b[i] = kx_P[i];
            }
            b[Nx] = kx_P[Nx] + kx_E[Nx];
            for (int i = 1; i <= Nx-1; i++) {
                c[i] = -kx_E[i];
            }
            for (int i = 1; i <= Nx; i++) {
                d[i] = RHS1[i][j];
                e[i] = RHS2[i][j];
            }
            tdma(Nx, a, b, c, d, e, x, y);

            for (int i = 1; i <= Nx; i++) {
                C1[i][j] = x[i];
                C2[i][j] = y[i];
            }
        }

        // calculate u_star
        for (int i = 1; i <= Nx; i++) {
            for (int j = 2; j <= Ny; j++) {
                a[j] = -ky_S[j];
            }
            b[1] = ky_S[1] + ky_P[1];
            for (int j = 2; j <= Ny-1; j++) {
                b[j] = ky_P[j];
            }
            b[Ny] = ky_P[Ny] + ky_N[Ny];
            for (int j = 1; j <= Ny-1; j++) {
                c[j] = -ky_N[j];
            }
            for (int j = 1; j <= Ny; j++) {
                d[j] = C1[i][j];
                e[j] = C2[i][j];
            }
            tdma(Ny, a, b, c, d, e, x, y);

            for (int j = 1; j <= Ny; j++) {
                u1_star[i][j] = x[j] + u1[i][j];
                u2_star[i][j] = y[j] + u2[i][j];
            }
        }

        // calculate u_tilde
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_tilde[i][j] = u1_star[i][j] + dt * (p[(i+1)][j] - p[(i-1)][j]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        // calculate U_star
        for (int i = 1; i <= Nx-1; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_star[i][j] = (u1_tilde[i][j]*dx[i+1] + u1_tilde[(i+1)][j]*dx[i]) / (dx[i] + dx[i+1])
                    - dt * (p[(i+1)][j] - p[i][j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                U2_star[i][j] = (u2_tilde[i][j]*dy[j+1] + u2_tilde[i][j+1]*dy[j]) / (dy[j] + dy[j+1])
                    - dt * (p[i][j+1] - p[i][j]) / (yc[j+1] - yc[j]);
            }
        }

        // calculate Q
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                Q[i][j] = 1 / (2.*Re) * ((U1_star[i][j] - U1_star[(i-1)][j]) / dx[i]
                                               + (U2_star[i][j] - U2_star[i][j-1]) / dy[j]);
            }
        }

        // solve for p_prime
        double res;
        for (int k = 1; k <= 1000000; k++) {
            res = 0;
            for (int i = 1; i <= Nx; i++) {
                p_prime[i][0] = p_prime[i][1];
                p_prime[i][Ny+1] = p_prime[i][Ny];
            }
            for (int j = 1; j <= Ny; j++) {
                p_prime[0][j] = p_prime[1][j];
                p_prime[(Nx+1)][j] = p_prime[Nx][j];
            }

            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (i == 1 && j == 1) {
                        continue;
                    }

                    double tmp = 1. / (kx_W[i] + kx_E[i] + ky_S[j] + ky_N[j]) * (
                        kx_W[i] * p_prime[(i-1)][j] + kx_E[i] * p_prime[(i+1)][j]
                        + ky_S[j] * p_prime[i][j-1] + ky_N[j] * p_prime[i][j+1] - Q[i][j]
                    );

                    if (isnan(tmp) || isinf(tmp)) {
                        fprintf(stderr, BOLD_TEXT RED_TEXT "ERROR: floating point error!\n");
                        exit(0);
                    }

                    res += fabs(tmp - p_prime[i][j]);
                    p_prime[i][j] = tmp;
                }
            }
            if (res / (Nx * Ny) <= 1e-9) {
                break;
            }
        }
        if (res / (Nx * Ny) > 1e-9) {
            printf(RED_TEXT "not converged!\n");
        }

        // calculate p_next
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                p_next[i][j] = p[i][j] + p_prime[i][j];
            }
        }

        // calculate u_next
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[i][j] = u1_star[i][j] - dt * (p_prime[(i+1)][j] - p_prime[(i-1)][j]) / (xc[i+1] - xc[i-1]);
                u2_next[i][j] = u2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        // calculate U_next
        for (int i = 1; i <= Nx-1; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_next[i][j] = U1_star[i][j] - dt * (p_prime[(i+1)][j] - p_prime[i][j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                U2_next[i][j] = U2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j]) / (yc[j+1] - yc[j]);
            }
        }

        // velocity bc
        for (int i = 1; i <= Nx; i++) {
            u1_next[i][0] = -u1_next[i][1];
            u1_next[i][Ny+1] = 2 - u1_next[i][Ny];
            u2_next[i][0] = -u2_next[i][1];
            u2_next[i][Ny+1] = 2 - u2_next[i][Ny];
        }
        for (int j = 1; j <= Ny; j++) {
            u1_next[0][j] = -u1_next[1][j];
            u1_next[(Nx+1)][j] = -u1_next[Nx][j];
            u2_next[0][j] = -u2_next[1][j];
            u2_next[(Nx+1)][j] = -u2_next[Nx][j];
        }
        for (int i = 0; i <= Nx; i++) {
            U1_next[i][0] = -U1_next[i][1];
            U1_next[i][Ny+1] = 2 - U1_next[i][Ny];
        }
        for (int j = 0; j <= Ny; j++) {
            U2_next[0][j] = -U2_next[1][j];
            U2_next[Nx+1][j] = -U2_next[Nx][j];
        }

        // update for next time step
        memcpy(p, p_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(u1, u1_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(u2, u2_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(U1, U1_next, sizeof(double)*(Nx+1)*(Ny+2));
        memcpy(U2, U2_next, sizeof(double)*(Nx+2)*(Ny+1));
        memcpy(N1_prev, N1, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(N2_prev, N2, sizeof(double)*(Nx+2)*(Ny+2));
        memset(p_prime, 0, sizeof(double)*(Nx+2)*(Ny+2));

        if (tstep % 5 == 0) {
            printf("tstep: %d\n", tstep);
        }
    }

    // calculate streamfunction
    for (int i = 1; i <= Nx-1; i++) {
        for (int j = 1; j <= Ny-1; j++) {
            psi[i][j] = psi[i][j-1] + dy[j] * U1[i][j];
        }
    }

    FILE *fp_out = fopen("cavity_result.txt", "w");
    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j <= Ny; j++) {
            fprintf(fp_out, "%.14lf ", psi[i][j]);
        }
        fprintf(fp_out, "\n");
    }
    fclose(fp_out);
}