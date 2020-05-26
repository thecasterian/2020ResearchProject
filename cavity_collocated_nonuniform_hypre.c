#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "HYPRE_struct_ls.h"

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

int main(int argc, char **argv) {
    /*===== Initialize program and parse arguments ===========================*/
    /* Initialize MPI */
    int myid, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Init();

    /* For the present, the number of process must be 1 */
    if (num_procs != 1) {
        if (myid == 0)
            printf("Must run with 1 processors!\n");
        MPI_Finalize();
        return 0;
    }

    /*===== Read input file ==================================================*/
    /* Input parameters */
    int Nx, Ny;
    double Re, dt;
    int numtstep;

    FILE *fp_in = fopen("cavity.in", "r");

    /* Read grid geometry */
    fscanf(fp_in, "%*s %d", &Nx);
    double xf[Nx+1];
    for (int i = 0; i <= Nx; i++) {
        fscanf(fp_in, "%lf", &xf[i]);
    }

    fscanf(fp_in, "%*s %d", &Ny);
    double yf[Ny+1];
    for (int j = 0; j <= Ny; j++) {
        fscanf(fp_in, "%lf", &yf[j]);
    }

    /* Read Re, dt, numtstep */
    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep);

    fclose(fp_in);

    /*===== Define variables =================================================*/
    /* Max. of Nx and Ny */
    const int Nm = Nx > Ny ? Nx : Ny;

    /* Grid variables */
    double xc[Nx+2], dx[Nx+2];
    double yc[Ny+2], dy[Ny+2];

    /* Pressure and velocities */
    double p[Nx+2][Ny+2], p_next[Nx+2][Ny+2], p_prime[Nx+2][Ny+2];
    double u1[Nx+2][Ny+2], u1_next[Nx+2][Ny+2], u1_star[Nx+2][Ny+2], u1_tilde[Nx+2][Ny+2];
    double u2[Nx+2][Ny+2], u2_next[Nx+2][Ny+2], u2_star[Nx+2][Ny+2], u2_tilde[Nx+2][Ny+2];

    double U1[Nx+1][Ny+2], U1_next[Nx+1][Ny+2], U1_star[Nx+1][Ny+2];
    double U2[Nx+2][Ny+1], U2_next[Nx+2][Ny+1], U2_star[Nx+2][Ny+1];

    /* Auxilary variables */
    double N1[Nx+2][Ny+2], N1_prev[Nx+2][Ny+2];
    double N2[Nx+2][Ny+2], N2_prev[Nx+2][Ny+2];
    double Q[Nx+2][Ny+2];
    double psi[Nx+1][Ny+1];

    /* For TDMA */
    double C1[Nx+2][Ny+2], C2[Nx+2][Ny+2], RHS1[Nx+2][Ny+2], RHS2[Nx+2][Ny+2];
    double kx_W[Nx+2], kx_P[Nx+2], kx_E[Nx+2];
    double ky_S[Ny+2], ky_P[Ny+2], ky_N[Ny+2];
    double a[Nm+2], b[Nm+2], c[Nm+2], d[Nm+2], e[Nm+2], x[Nm+2], y[Nm+2];

    /* For HYPRE */
    HYPRE_StructGrid grid;
    HYPRE_StructStencil stencil;
    HYPRE_StructMatrix matrix;
    HYPRE_StructVector rhsvec;
    HYPRE_StructVector resvec;
    HYPRE_StructSolver structsolver;

    /*===== Calculate grid variables =========================================*/
    for (int i = 1; i <= Nx; i++) {
        dx[i] = xf[i] - xf[i-1];
        xc[i] = (xf[i] + xf[i-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        dy[j] = yf[j] - yf[j-1];
        yc[j] = (yf[j] + yf[j-1]) / 2;
    }

    /* Ghost cells */
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

    /*===== Initialize HYPRE variables =======================================*/
    /* Setup grid */
    {
        HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

        int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny};
        HYPRE_StructGridSetExtents(grid, ilower, iupper);

        HYPRE_StructGridAssemble(grid);
    }
    /* Setup stencil */
    {
        HYPRE_StructStencilCreate(2, 5, &stencil);

        /* Stencil offset: center, up, right, down, left */
        int offsets[5][2] = {{0, 0}, {0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        for (int i = 0; i < 5; i++)
            HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
    }
    /* Setup martix */
    {
        HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &matrix);
        HYPRE_StructMatrixInitialize(matrix);

        /* Setup coefficients */
        {
            int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny};
            int stencil_indices[5] = {0, 1, 2, 3, 4};
            /* Nx*Ny grid points, each with 5 stencil points */
            double values[5*Nx*Ny];

            int m = 0;
            for (int j = 1; j <= Ny; j++) {
                for (int i = 1; i <= Nx; i++) {
                    values[5*m] = kx_W[i] + kx_E[i] + ky_S[j] + ky_N[j];
                    values[5*m+1] = -ky_N[j];
                    values[5*m+2] = -kx_E[i];
                    values[5*m+3] = -ky_S[j];
                    values[5*m+4] = -kx_W[i];
                    m++;
                }
            }

            HYPRE_StructMatrixSetBoxValues(matrix, ilower, iupper, 5, stencil_indices, values);
        }

        /* Setup coefficients of boundary cells */
        {
            double values[2*Nm+4];

            /* Upper boundary */
            {
                int ilower[2] = {1, Ny}, iupper[2] = {Nx, Ny};
                int stencil_indices[2] = {0, 1};

                for (int i = 1; i <= Nx; i++) {
                    values[2*i] = kx_W[i] + kx_E[i] + ky_S[Ny];
                    values[2*i+1] = 0;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilower, iupper, 2, stencil_indices, values);
            }
            /* Right boundary */
            {
                int ilower[2] = {Nx, 1}, iupper[2] = {Nx, Ny};
                int stencil_indices[2] = {0, 2};

                for (int j = 1; j <= Ny; j++) {
                    values[2*j] = kx_W[Nx] + ky_S[j] + ky_N[j];
                    values[2*j+1] = 0;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilower, iupper, 2, stencil_indices, values);
            }
            /* Lower boundary */
            {
                int ilower[2] = {1, 1}, iupper[2] = {Nx, 1};
                int stencil_indices[2] = {0, 3};

                for (int i = 1; i <= Nx; i++) {
                    values[2*i] = kx_W[i] + kx_E[i] + ky_N[1];
                    values[2*i+1] = 0;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilower, iupper, 2, stencil_indices, values);
            }
            /* Left boundary */
            {
                int ilower[2] = {1, 1}, iupper[2] = {1, Ny};
                int stencil_indices[2] = {0, 4};

                for (int j = 1; j <= Ny; j++) {
                    values[2*j] = kx_E[1] + ky_S[j] + ky_N[j];
                    values[2*j+1] = 0;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilower, iupper, 2, stencil_indices, values);
            }
            /* Upper left corner */
            {
                int i[2] = {1, Ny};
                int stencil_indices[1] = {0};

                values[0] = kx_E[1] + ky_S[Ny];

                HYPRE_StructMatrixSetBoxValues(matrix, i, i, 1, stencil_indices, values);
            }
            /* Upper right corner */
            {
                int i[2] = {Nx, Ny};
                int stencil_indices[1] = {0};

                values[0] = kx_W[Nx] + ky_S[Ny];

                HYPRE_StructMatrixSetBoxValues(matrix, i, i, 1, stencil_indices, values);
            }
            /* Lower left corner */
            {
                int i[2] = {1, 1};
                int stencil_indices[5] = {0, 1, 2, 3, 4};

                values[0] = 1;
                for (int k = 1; k <= 4; k++)
                    values[k] = 0;

                HYPRE_StructMatrixSetBoxValues(matrix, i, i, 5, stencil_indices, values);
            }
            /* Lower right corner */
            {
                int i[2] = {Nx, 1};
                int stencil_indices[1] = {0};

                values[0] = kx_W[Nx] + ky_N[1];

                HYPRE_StructMatrixSetBoxValues(matrix, i, i, 1, stencil_indices, values);
            }
        }

        HYPRE_StructMatrixAssemble(matrix);
    }
    /* Initialize vector */
    {
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &rhsvec);
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &resvec);
        HYPRE_StructVectorInitialize(rhsvec);
        HYPRE_StructVectorInitialize(resvec);
    }
    /* Setup solver */
    {
        /* Create an empty PCG Struct solver */
        HYPRE_StructPCGCreate(MPI_COMM_WORLD, &structsolver);

        /* Convergence tolerance */
        HYPRE_StructPCGSetTol(structsolver, 1.e-6);
        /* Info. level */
        HYPRE_StructPCGSetPrintLevel(structsolver, 0);
    }

    /*===== Initialize flow ==================================================*/
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
    for (int i = 0; i <= Nx; i++) {
        U1[i][Ny+1] = 2;
    }
    for (int i = 1; i <= Nx; i++) {
        u1[i][Ny+1] = 2;
    }
    memcpy(N1_prev, N1, sizeof(double)*(Nx+2)*(Ny+2));
    memcpy(N2_prev, N2, sizeof(double)*(Nx+2)*(Ny+2));

    /*===== Main loop ========================================================*/
    for (int tstep = 1; tstep <= numtstep; tstep++) {
        /* Calculate N */
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

        /* Calculate RHS */
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

        /* Calcuate C */
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

        /* Calculate u_star */
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

        /* Calculate u_tilde */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_tilde[i][j] = u1_star[i][j] + dt * (p[(i+1)][j] - p[(i-1)][j]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        /* Calculate U_star */
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

        /* Calculate Q */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                Q[i][j] = 1 / (2.*Re) * ((U1_star[i][j] - U1_star[(i-1)][j]) / dx[i]
                                               + (U2_star[i][j] - U2_star[i][j-1]) / dy[j]);
            }
        }

        /* Calculate p_prime */
        {
            int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny};
            double values[Nx*Ny];

            int m = 0;
            for (int j = 1; j <= Ny; j++) {
                for (int i = 1; i <= Nx; i++) {
                    if (i == 1 && j == 1) {
                        values[m] = 0;
                    }
                    else {
                        values[m] = -Q[i][j];
                    }
                    m++;
                }
            }

            HYPRE_StructVectorSetBoxValues(rhsvec, ilower, iupper, values);

            for (int i = 0; i < Nx*Ny; i++)
                values[i] = 0;
            HYPRE_StructVectorSetBoxValues(resvec, ilower, iupper, values);

            HYPRE_StructVectorAssemble(rhsvec);
            HYPRE_StructVectorAssemble(resvec);

            HYPRE_StructPCGSetup(structsolver, matrix, rhsvec, resvec);
            HYPRE_StructPCGSolve(structsolver, matrix, rhsvec, resvec);

            HYPRE_StructVectorGetBoxValues(resvec, ilower, iupper, values);
            m = 0;
            for (int j = 1; j <= Ny; j++) {
                for (int i = 1; i <= Nx; i++) {
                    p_prime[i][j] = values[m];
                    m++;
                }
            }
        }

        /* Calculate p_next */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                p_next[i][j] = p[i][j] + p_prime[i][j];
            }
        }

        /* Calculate u_next */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[i][j] = u1_star[i][j] - dt * (p_prime[(i+1)][j] - p_prime[(i-1)][j]) / (xc[i+1] - xc[i-1]);
                u2_next[i][j] = u2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        /* Calculate U_next */
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

        /* Set velocity boundary conditions */
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

        /* Update for next time step */
        memcpy(p, p_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(u1, u1_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(u2, u2_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(U1, U1_next, sizeof(double)*(Nx+1)*(Ny+2));
        memcpy(U2, U2_next, sizeof(double)*(Nx+2)*(Ny+1));
        memcpy(N1_prev, N1, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(N2_prev, N2, sizeof(double)*(Nx+2)*(Ny+2));
        memset(p_prime, 0, sizeof(double)*(Nx+2)*(Ny+2));

        if (tstep % 50 == 0) {
            printf("tstep: %d\n", tstep);
        }
    }

    /*===== Export result ====================================================*/
    /* Calculate streamfunction */
    for (int i = 1; i <= Nx-1; i++) {
        for (int j = 1; j <= Ny-1; j++) {
            psi[i][j] = psi[i][j-1] + dy[j] * U1[i][j];
        }
    }

    /* Write to output file */
    FILE *fp_out = fopen("result/cavity_result.txt", "w");
    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j <= Ny; j++) {
            fprintf(fp_out, "%.14lf ", psi[i][j]);
        }
        fprintf(fp_out, "\n");
    }
    fclose(fp_out);

    /* Free memory */
    HYPRE_StructGridDestroy(grid);
    HYPRE_StructStencilDestroy(stencil);
    HYPRE_StructMatrixDestroy(matrix);
    HYPRE_StructVectorDestroy(rhsvec);
    HYPRE_StructVectorDestroy(resvec);
    HYPRE_StructPCGDestroy(structsolver);

    /* Finalize Hypre */
    HYPRE_Finalize();

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}
