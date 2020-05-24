#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "HYPRE_struct_ls.h"
#include "_hypre_struct_mv.h"

#define Re 400
#define dt 0.01

#define N 64
#define numtstep 1000

// for mpi
int myid, num_procs;

// for hypre
HYPRE_StructGrid     grid;
HYPRE_StructStencil  stencil;
HYPRE_StructMatrix   matrix_A;
HYPRE_StructVector   vector_b;
HYPRE_StructVector   vector_x;
HYPRE_StructSolver   solver;

typedef double mat[N+2][N+2];
typedef double mat1[N+1][N+2];
typedef double mat2[N+2][N+1];

mat p, p_next, p_prime;
mat u1, u1_next, u1_star, u1_tilde;
mat u2, u2_next, u2_star, u2_tilde;
mat1 U1, U1_next, U1_star;
mat2 U2, U2_next, U2_star;

mat N1, N1_prev;
mat N2, N2_prev;
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

void setup_hypre_solver(void);
void calc_N(void);
void calc_RHS(void);
void calc_C(void);
void calc_u_star(void);
void calc_U_star(void);
void solve_p_prime(void);

int main(int argc, char *argv[]) {
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Init();

    if (num_procs != 1) {
        if (myid == 0)
            printf("Must run with 1 processors!\n");
        MPI_Finalize();
        return 0;
    }

    // initialize stencil, matrix, and vector
    setup_hypre_solver();

    // initialize
    for (int i = 0; i <= N; i++) {
        U1[i][N+1] = 2;
    }
    for (int i = 1; i <= N; i++) {
        u1[i][N+1] = 2;
    }

    for (int tstep = 1; tstep <= numtstep; tstep++) {
        // calculate N
        calc_N();

        // calculate RHS
        calc_RHS();

        // calcuate C
        calc_C();

        // calculate u_star
        calc_u_star();

        // calculate u_tilde
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                u1_tilde[i][j] = u1_star[i][j] + dt * (p[i+1][j]-p[i-1][j]) / (2*h);
                u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1]-p[i][j-1]) / (2*h);
            }
        }

        // calculate U_star
        calc_U_star();

        // calculate Q
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                Q[i][j] = (U1_star[i][j]-U1_star[i-1][j] + U2_star[i][j]-U2_star[i][j-1]) / (dt*h);
            }
        }

        // solve for p_prime
        solve_p_prime();

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

    /* Free memory */
    HYPRE_StructGridDestroy(grid);
    HYPRE_StructStencilDestroy(stencil);
    HYPRE_StructMatrixDestroy(matrix_A);
    HYPRE_StructVectorDestroy(vector_b);
    HYPRE_StructVectorDestroy(vector_x);
    HYPRE_StructPCGDestroy(solver);

    /* Finalize Hypre */
    HYPRE_Finalize();

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}

void calc_N(void) {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            N1[i][j] = (U1[i][j] * (u1[i+1][j]+u1[i][j])/2 - U1[i-1][j] * (u1[i][j]+u1[i-1][j])/2) / h
                + (U2[i][j] * (u1[i][j+1]+u1[i][j])/2 - U2[i][j-1] * (u1[i][j]+u1[i][j-1])/2) / h;
            N2[i][j] = (U1[i][j] * (u2[i+1][j]+u2[i][j])/2 - U1[i-1][j] * (u2[i][j]+u2[i-1][j])/2) / h
                + (U2[i][j] * (u2[i][j+1]+u2[i][j])/2 - U2[i][j-1] * (u2[i][j]+u2[i][j-1])/2) / h;
        }
    }
}

void calc_RHS(void) {
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
}

void calc_C(void) {
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
}

void calc_u_star(void) {
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
}

void calc_U_star(void) {
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
}

void setup_hypre_solver(void) {
    {
        HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

        int ilower[2] = {1, 1}, iupper[2] = {N, N};
        HYPRE_StructGridSetExtents(grid, ilower, iupper);

        HYPRE_StructGridAssemble(grid);
    }
    {
        HYPRE_StructStencilCreate(2, 5, &stencil);

        /* Stencil offset: center, up, right, down, left */
        int offsets[5][2] = {{0, 0}, {0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        for (int i = 0; i < 5; i++)
            HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
    }
    {
        HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &matrix_A);
        HYPRE_StructMatrixInitialize(matrix_A);

        {
            int ilower[2] = {1, 1}, iupper[2] = {N, N};
            int stencil_indices[5] = {0, 1, 2, 3, 4};
            /* N*N grid points, each with 5 stencil entries */
            double *values = malloc(sizeof(double) * N*N*5);

            for (int i = 0; i < N*N*5; i += 5)
            {
                values[i] = 4.;
                for (int j = 1; j < 5; j++)
                values[i+j] = -1.;
            }
            HYPRE_StructMatrixSetBoxValues(matrix_A, ilower, iupper,
                                           5, stencil_indices, values);

            free(values);
        }
        /* Set the coefficients of boundary grid points */
        {
            double values[2*N];
            for (int i = 0; i < 2*N; i += 2)
            {
                values[i] = 3.;
                values[i+1] = 0.;
            }
            {
                /* upper boundary */
                int ilower[2] = {1, N}, iupper[2] = {N, N};
                int stencil_indices[2] = {0, 1};
                HYPRE_StructMatrixSetBoxValues(matrix_A, ilower, iupper,
                                               2, stencil_indices, values);
            }
            {
                /* right boundary */
                int ilower[2] = {N, 1}, iupper[2] = {N, N};
                int stencil_indices[2] = {0, 2};
                HYPRE_StructMatrixSetBoxValues(matrix_A, ilower, iupper,
                                               2, stencil_indices, values);
            }
            {
                /* lower boundary */
                int ilower[2] = {1, 1}, iupper[2] = {N, 1};
                int stencil_indices[2] = {0, 3};
                HYPRE_StructMatrixSetBoxValues(matrix_A, ilower, iupper,
                                               2, stencil_indices, values);
            }
            {
                /* left boundary */
                int ilower[2] = {1, 1}, iupper[2] = {1, N};
                int stencil_indices[2] = {0, 4};
                HYPRE_StructMatrixSetBoxValues(matrix_A, ilower, iupper,
                                               2, stencil_indices, values);
            }
            values[0] = 2.;
            {
                /* upper left corner */
                int i[2] = {1, N};
                int stencil_indices[1] = {0};
                HYPRE_StructMatrixSetBoxValues(matrix_A, i, i,
                                               1, stencil_indices, values);
            }
            {
                /* upper right corner */
                int i[2] = {N, N};
                int stencil_indices[1] = {0};
                HYPRE_StructMatrixSetBoxValues(matrix_A, i, i,
                                               1, stencil_indices, values);
            }
            {
                /* lower right corner */
                int i[2] = {N, 1};
                int stencil_indices[1] = {0};
                HYPRE_StructMatrixSetBoxValues(matrix_A, i, i,
                                               1, stencil_indices, values);
            }
            values[0] = 1.;
            for (int i = 1; i <= 4; i++)
                values[i] = 0.;
            {
                /* lower left corner */
                int i[2] = {1, 1};
                int stencil_indices[5] = {0, 1, 2, 3, 4};
                HYPRE_StructMatrixSetBoxValues(matrix_A, i, i,
                                               5, stencil_indices, values);
            }
        }

        HYPRE_StructMatrixAssemble(matrix_A);
    }
    {
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &vector_b);
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &vector_x);
        HYPRE_StructVectorInitialize(vector_b);
        HYPRE_StructVectorInitialize(vector_x);
    }
}

void solve_p_prime(void) {
    int ilower[2] = {1, 1}, iupper[2] = {N, N};
    double values[N*N];

    int m = 0;
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= N; i++) {
            if (i == 1 && j == 1)
                values[m] = 0.;
            else
                values[m] = -h*h*Q[i][j];
            m++;
        }
    HYPRE_StructVectorSetBoxValues(vector_b, ilower, iupper, values);

    for (int i = 0; i < N*N; i++)
        values[i] = 0.;
    HYPRE_StructVectorSetBoxValues(vector_x, ilower, iupper, values);

    HYPRE_StructVectorAssemble(vector_b);
    HYPRE_StructVectorAssemble(vector_x);

    /* Create an empty PCG Struct solver */
    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

    /* Set some parameters */
    HYPRE_StructPCGSetTol(solver, 1.e-6); /* convergence tolerance */
    HYPRE_StructPCGSetPrintLevel(solver, 0); /* amount of info. printed */

    /* Setup and solve */
    HYPRE_StructPCGSetup(solver, matrix_A, vector_b, vector_x);
    HYPRE_StructPCGSolve(solver, matrix_A, vector_b, vector_x);

    /* Copy result to p_prime */
    HYPRE_StructVectorGetBoxValues(vector_x, ilower, iupper, values);
    m = 0;
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= N; i++) {
            p_prime[i][j] = values[m];
            m++;
        }
}