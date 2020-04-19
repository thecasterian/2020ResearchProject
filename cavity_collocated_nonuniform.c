#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// input variables
int Nx, Ny;
double *xf, *yf;                                // Nx+1, Ny+1
double Re;
double dt;
int numtstep;

// grid geometry
double *xc, *dx;                                // Nx+2
double *yc, *dy;                                // Ny+2

// pressure and velocity
double *p, *p_next, *p_prime;                   // (Nx+2) x (Ny+2)
double *u1, *u1_next, *u1_star, *u1_tilde;      // (Nx+2) x (Ny+2)
double *u2, *u2_next, *u2_star, *u2_tilde;      // (Nx+2) x (Ny+2)
double *U1, *U1_next, *U1_star;                 // (Nx+1) x (Ny+2)
double *U2, *U2_next, *U2_star;                 // (Nx+2) x (Ny+1)

// auxiliary
double *N1, *N1_prev;                           // (Nx+2) x (Ny+2)
double *N2, *N2_prev;                           // (Nx+2) x (Ny+2)
double *Q;                                      // (Nx+2) x (Ny+2)
double *psi;                                    // (Nx+1) x (Ny+1)

// for tdma
double *C1, *C2, *RHS1, *RHS2;                  // (Nx+2) x (Ny+2)
double *kx_W, *kx_P, *kx_E;                     // Nx+2
double *ky_S, *ky_P, *ky_N;                     // Ny+2
double *a, *b, *c, *d, *e, *x, *y;              // max(Nx, Ny)+2

// set of arrays
double **arr0[] = {                             // arrays with size of (Nx+2) x (Ny+2)
    &p, &p_next, &p_prime,
    &u1, &u1_next, &u1_star, &u1_tilde,
    &u2, &u2_next, &u2_star, &u2_tilde,
    &N1, &N1_prev,
    &N2, &N2_prev,
    &Q,
    &C1, &C2, &RHS1, &RHS2
};
double **arr1[] = {                             // arrays with size of (Nx+1) x (Ny+2)
    &U1, &U1_next, &U1_star
};
double **arr2[] = {                             // arrays with size of (Nx+2) x (Ny+1)
    &U2, &U2_next, &U2_star
};
double **arr_tdma[] = {                         // arrays with size of max(Nx, Ny)+2
    &a, &b, &c, &d, &e, &x, &y
};

double *falloc(size_t size) {
    return (double *)calloc(sizeof(double), size);
}

double tdma(int n) {
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

void calc_N(void) {
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            double u1_e = (u1[i*(Ny+2)+j] * dx[i+1] + u1[(i+1)*(Ny+2)+j] * dx[i]) / (dx[i] + dx[i+1]);
            double u2_e = (u2[i*(Ny+2)+j] * dx[i+1] + u2[(i+1)*(Ny+2)+j] * dx[i]) / (dx[i] + dx[i+1]);
            double u1_w = (u1[(i-1)*(Ny+2)+j] * dx[i] + u1[i*(Ny+2)+j] * dx[i-1]) / (dx[i-1] + dx[i]);
            double u2_w = (u2[(i-1)*(Ny+2)+j] * dx[i] + u2[i*(Ny+2)+j] * dx[i-1]) / (dx[i-1] + dx[i]);
            double u1_n = (u1[i*(Ny+2)+j] * dy[j+1] + u1[i*(Ny+2)+j+1] * dy[j]) / (dy[j] + dy[j+1]);
            double u2_n = (u2[i*(Ny+2)+j] * dy[j+1] + u2[i*(Ny+2)+j+1] * dy[j]) / (dy[j] + dy[j+1]);
            double u1_s = (u1[i*(Ny+2)+j-1] * dy[j] + u1[i*(Ny+2)+j] * dy[j-1]) / (dy[j-1] + dy[j]);
            double u2_s = (u2[i*(Ny+2)+j-1] * dy[j] + u2[i*(Ny+2)+j] * dy[j-1]) / (dy[j-1] + dy[j]);

            N1[i*(Ny+2)+j] = (U1[i*(Ny+2)+j]*u1_e - U1[(i-1)*(Ny+2)+j]*u1_w) / dx[i] + (U2[i*(Ny+1)+j]*u1_n - U2[i*(Ny+1)+j-1]*u1_s) / dy[j];
            N2[i*(Ny+2)+j] = (U1[i*(Ny+2)+j]*u2_e - U1[(i-1)*(Ny+2)+j]*u2_w) / dx[i] + (U2[i*(Ny+1)+j]*u2_n - U2[i*(Ny+1)+j-1]*u2_s) / dy[j];
        }
    }
}

int main(void) {
    FILE *fp_in = fopen("cavity.in", "r");

    // read inputs
    fscanf(fp_in, "%*s %d", &Nx);
    xf = malloc(sizeof(double) * (Nx+1));
    for (int i = 0; i <= Nx; i++) {
        fscanf(fp_in, "%lf", &xf[i]);
    }
    fscanf(fp_in, "%*s %d", &Ny);
    yf = malloc(sizeof(double) * (Ny+1));
    for (int j = 0; j <= Ny; j++) {
        fscanf(fp_in, "%lf", &yf[j]);
    }

    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep);

    fclose(fp_in);

    // allocate arrays
    xc = falloc(Nx+2);
    dx = falloc(Nx+2);
    yc = falloc(Ny+2);
    dy = falloc(Ny+2);
    for (int k = 0; k < sizeof(arr0)/sizeof(double **); k++) {
        *(arr0[k]) = falloc((Nx+2) * (Ny+2));
    }
    for (int k = 0; k < sizeof(arr1)/sizeof(double **); k++) {
        *(arr1[k]) = falloc((Nx+1) * (Ny+2));
    }
    for (int k = 0; k < sizeof(arr2)/sizeof(double **); k++) {
        *(arr2[k]) = falloc((Nx+2) * (Ny+1));
    }
    psi = falloc((Nx+1) * (Ny+1));
    kx_W = falloc(Nx+2);
    kx_P = falloc(Nx+2);
    kx_E = falloc(Nx+2);
    ky_S = falloc(Ny+2);
    ky_P = falloc(Ny+2);
    ky_N = falloc(Ny+2);
    for (int k = 0; k < sizeof(arr_tdma)/sizeof(double **); k++) {
        *(arr_tdma[k]) = falloc((Nx > Ny ? Nx : Ny) + 2);
    }

    // calculate grid variables
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

    // tdma coefficients
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
        U1[i*(Ny+2)+Ny+1] = 2;
    }
    for (int i = 1; i <= Nx; i++) {
        u1[i*(Ny+2)+Ny+1] = 2;
    }
    calc_N();
    memcpy(N1_prev, N1, sizeof(double)*(Nx+2)*(Ny+2));
    memcpy(N2_prev, N2, sizeof(double)*(Nx+2)*(Ny+2));

    for (int tstep = 1; tstep <= numtstep; tstep++) {
        // calculate N
        calc_N();

        // calculate RHS
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                RHS1[i*(Ny+2)+j] = -dt/2 * (3*N1[i*(Ny+2)+j] - N1_prev[i*(Ny+2)+j])
                    - dt * (p[(i+1)*(Ny+2)+j] - p[(i-1)*(Ny+2)+j]) / (xc[i+1] - xc[i-1])
                    + 2 * (kx_W[i]*u1[(i-1)*(Ny+2)+j] + kx_E[i]*u1[(i+1)*(Ny+2)+j]
                           + ky_S[j]*u1[i*(Ny+2)+j-1] + ky_N[j]*u1[i*(Ny+2)+j+1]
                           - (kx_W[i]+kx_E[i]+ky_S[j]+ky_N[j])*u1[i*(Ny+2)+j]);
                RHS2[i*(Ny+2)+j] = -dt/2 * (3*N2[i*(Ny+2)+j] - N2_prev[i*(Ny+2)+j])
                    - dt * (p[i*(Ny+2)+j+1] - p[i*(Ny+2)+j-1]) / (yc[j+1] - yc[j-1])
                    + 2 * (kx_W[i]*u2[(i-1)*(Ny+2)+j] + kx_E[i]*u2[(i+1)*(Ny+2)+j]
                           + ky_S[j]*u2[i*(Ny+2)+j-1] + ky_N[j]*u2[i*(Ny+2)+j+1]
                           - (kx_W[i]+kx_E[i]+ky_S[j]+ky_N[j])*u2[i*(Ny+2)+j]);
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
                d[i] = RHS1[i*(Ny+2)+j];
                e[i] = RHS2[i*(Ny+2)+j];
            }
            tdma(Nx);

            for (int i = 1; i <= Nx; i++) {
                C1[i*(Ny+2)+j] = x[i];
                C2[i*(Ny+2)+j] = y[i];
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
                d[j] = C1[i*(Ny+2)+j];
                e[j] = C2[i*(Ny+2)+j];
            }
            tdma(Ny);

            for (int j = 1; j <= Ny; j++) {
                u1_star[i*(Ny+2)+j] = x[j] + u1[i*(Ny+2)+j];
                u2_star[i*(Ny+2)+j] = y[j] + u2[i*(Ny+2)+j];
            }
        }

        // calculate u_tilde
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_tilde[i*(Ny+2)+j] = u1_star[i*(Ny+2)+j] + dt * (p[(i+1)*(Ny+2)+j] - p[(i-1)*(Ny+2)+j]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i*(Ny+2)+j] = u2_star[i*(Ny+2)+j] + dt * (p[i*(Ny+2)+j+1] - p[i*(Ny+2)+j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        // calculate U_star
        for (int i = 1; i <= Nx-1; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_star[i*(Ny+2)+j] = (u1_tilde[i*(Ny+2)+j]*dx[i+1] + u1_tilde[(i+1)*(Ny+2)+j]*dx[i]) / (dx[i] + dx[i+1])
                    - dt * (p[(i+1)*(Ny+2)+j] - p[i*(Ny+2)+j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                U2_star[i*(Ny+1)+j] = (u2_tilde[i*(Ny+2)+j]*dy[j+1] + u2_tilde[i*(Ny+2)+j+1]*dy[j]) / (dy[j] + dy[j+1])
                    - dt * (p[i*(Ny+2)+j+1] - p[i*(Ny+2)+j]) / (yc[j+1] - yc[j]);
            }
        }

        // calculate Q
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                Q[i*(Ny+2)+j] = 1 / (2.*Re) * ((U1_star[i*(Ny+2)+j] - U1_star[(i-1)*(Ny+2)+j]) / dx[i]
                                               + (U2_star[i*(Ny+1)+j] - U2_star[i*(Ny+1)+j-1]) / dy[j]);
            }
        }

        // solve for p_prime
        double res;
        for (int k = 1; k <= 1000000; k++) {
            res = 0;
            for (int i = 1; i <= Nx; i++) {
                p_prime[i*(Ny+2)+0] = p_prime[i*(Ny+2)+1];
                p_prime[i*(Ny+2)+Ny+1] = p_prime[i*(Ny+2)+Ny];
            }
            for (int j = 1; j <= Ny; j++) {
                p_prime[0*(Ny+2)+j] = p_prime[1*(Ny+2)+j];
                p_prime[(Nx+1)*(Ny+2)+j] = p_prime[Nx*(Ny+2)+j];
            }

            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (i == 1 && j == 1) {
                        continue;
                    }
                    double tmp = 1. / (kx_W[i] + kx_E[i] + ky_S[j] + ky_N[j]) * (
                        kx_W[i] * p_prime[(i-1)*(Ny+2)+j] + kx_E[i] * p_prime[(i+1)*(Ny+2)+j]
                        + ky_S[j] * p_prime[i*(Ny+2)+j-1] + ky_N[j] * p_prime[i*(Ny+2)+j+1] - Q[i*(Ny+2)+j]
                    );
                    res += fabs(tmp - p_prime[i*(Ny+2)+j]);
                    p_prime[i*(Ny+2)+j] = tmp;
                }
            }
            if (res / (Nx * Ny) <= 1e-9) {
                break;
            }
        }
        if (res / (Nx * Ny) > 1e-9) {
            printf("not converged!\n");
        }

        // calculate p_next
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                p_next[i*(Ny+2)+j] = p[i*(Ny+2)+j] + p_prime[i*(Ny+2)+j];
            }
        }

        // calculate u_next
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[i*(Ny+2)+j] = u1_star[i*(Ny+2)+j] - dt * (p_prime[(i+1)*(Ny+2)+j] - p_prime[(i-1)*(Ny+2)+j]) / (xc[i+1] - xc[i-1]);
                u2_next[i*(Ny+2)+j] = u2_star[i*(Ny+2)+j] - dt * (p_prime[i*(Ny+2)+j+1] - p_prime[i*(Ny+2)+j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        // calculate U_next
        for (int i = 1; i <= Nx-1; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_next[i*(Ny+2)+j] = U1_star[i*(Ny+2)+j] - dt * (p_prime[(i+1)*(Ny+2)+j] - p_prime[i*(Ny+2)+j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                U2_next[i*(Ny+1)+j] = U2_star[i*(Ny+1)+j] - dt * (p_prime[i*(Ny+2)+j+1] - p_prime[i*(Ny+2)+j]) / (yc[j+1] - yc[j]);
            }
        }

        // velocity bc
        for (int i = 1; i <= Nx; i++) {
            u1_next[i*(Ny+2)+0] = -u1_next[i*(Ny+2)+1];
            u1_next[i*(Ny+2)+Ny+1] = 2 - u1_next[i*(Ny+2)+Ny];
            u2_next[i*(Ny+2)+0] = -u2_next[i*(Ny+2)+1];
            u2_next[i*(Ny+2)+Ny+1] = 2 - u2_next[i*(Ny+2)+Ny];
        }
        for (int j = 1; j <= Ny; j++) {
            u1_next[0*(Ny+2)+j] = -u1_next[1*(Ny+2)+j];
            u1_next[(Nx+1)*(Ny+2)+j] = -u1_next[Nx*(Ny+2)+j];
            u2_next[0*(Ny+2)+j] = -u2_next[1*(Ny+2)+j];
            u2_next[(Nx+1)*(Ny+2)+j] = -u2_next[Nx*(Ny+2)+j];
        }
        for (int i = 0; i <= Nx; i++) {
            U1_next[i*(Ny+2)+0] = -U1_next[i*(Ny+2)+1];
            U1_next[i*(Ny+2)+Ny+1] = 2 - U1_next[i*(Ny+2)+Ny];
        }
        for (int j = 0; j <= Ny; j++) {
            U2_next[0*(Ny+1)+j] = -U2_next[1*(Ny+1)+j];
            U2_next[Nx+1*(Ny+1)+j] = -U2_next[Nx*(Ny+1)+j];
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
            psi[i*(Ny+1)+j] = psi[i*(Ny+1)+j-1] + dy[j] * U1[i*(Ny+2)+j];
        }
    }

    FILE *fp_out = fopen("cavity_result.txt", "w");
    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j <= Ny; j++) {
            fprintf(fp_out, "%.14lf ", psi[i*(Ny+1)+j]);
        }
        fprintf(fp_out, "\n");
    }
    fclose(fp_out);
}