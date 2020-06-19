#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "HYPRE_struct_ls.h"

#define DEBUG_TICTOC 0

#define ASSERT_ABORT(expr) \
do { \
    if (!(expr)) { \
        MPI_Abort(MPI_COMM_WORLD, 0); \
    } \
} while (0)

#define VA_GENERIC( \
_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,  x, ... \
) x
#define AT_1(a) (a)
#define AT_2(a, ...) ((a) && AT_1(__VA_ARGS__))
#define AT_3(a, ...) ((a) && AT_2(__VA_ARGS__))
#define AT_4(a, ...) ((a) && AT_3(__VA_ARGS__))
#define AT_5(a, ...) ((a) && AT_4(__VA_ARGS__))
#define AT_6(a, ...) ((a) && AT_5(__VA_ARGS__))
#define AT_7(a, ...) ((a) && AT_6(__VA_ARGS__))
#define AT_8(a, ...) ((a) && AT_7(__VA_ARGS__))
#define AT_9(a, ...) ((a) && AT_8(__VA_ARGS__))
#define AT_10(a, ...) ((a) && AT_9(__VA_ARGS__))
#define AT_11(a, ...) ((a) && AT_10(__VA_ARGS__))
#define AT_12(a, ...) ((a) && AT_11(__VA_ARGS__))
#define AT_13(a, ...) ((a) && AT_12(__VA_ARGS__))
#define AT_14(a, ...) ((a) && AT_13(__VA_ARGS__))
#define AT_15(a, ...) ((a) && AT_14(__VA_ARGS__))
#define AT_16(a, ...) ((a) && AT_15(__VA_ARGS__))

#define ALL_TRUE(...) VA_GENERIC( \
__VA_ARGS__, \
AT_16, AT_15, AT_14, AT_13, AT_12, AT_11, AT_10, AT_9, \
AT_8, AT_7, AT_6, AT_5, AT_4, AT_3, AT_2, AT_1 \
)(__VA_ARGS__)
#define CHECK_ALLOC_SUCCESS(...) \
do { \
    if (!ALL_TRUE(__VA_ARGS__)) { \
        fprintf(stderr, "error: failed to allocate memory\n"); \
        MPI_Abort(MPI_COMM_WORLD, 0); \
    } \
} while (0)

void forward(double *a, double *b, double *c, double *d, double *e, int start, int end) {
    for (int k = start+1; k <= end; k++) {
        double m = a[k] / b[k-1];
        b[k] -= m * c[k-1];
        d[k] -= m * d[k-1];
        e[k] -= m * e[k-1];
    }
}

void back(double *b, double *c, double *d, double *e, double *x, double *y, int start, int end) {
    for (int k = end-1; k >= start; k--) {
        x[k] = (d[k] - c[k] * x[k+1]) / b[k];
        y[k] = (e[k] - c[k] * y[k+1]) / b[k];
    }
}

int myid, num_procs;
struct timespec start_time, end_time;
struct timespec start_time_total, end_time_total;

void tic() {
#if DEBUG_TICTOC
    if (myid == 0) {
        clock_gettime(CLOCK_REALTIME, &start_time);
    }
#endif
}

void toc(const char *const msg) {
#if DEBUG_TICTOC
    if (myid == 0) {
        clock_gettime(CLOCK_REALTIME, &end_time);
        double elapsed_time = (end_time.tv_sec-start_time.tv_sec) + (end_time.tv_nsec-start_time.tv_nsec)/1.e9;
        printf("%-20s elapsed time (s): %.6lf\n", msg, elapsed_time);
    }
#endif
}

int main(int argc, char **argv) {
    /*===== Initialize program and parse arguments ===========================*/
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Init();

    /*===== Read input file ==================================================*/
    /* Input parameters */
    int Nx_, Ny_;
    double Re_, dt_;
    int numtstep_;

    FILE *fp_in = fopen("cavity.in", "r");
    ASSERT_ABORT(fp_in != 0);

    /* Read grid geometry */
    ASSERT_ABORT(fscanf(fp_in, "%*s %d", &Nx_) == 1);
    double *const xf = calloc(Nx_+1, sizeof(double));
    CHECK_ALLOC_SUCCESS(xf);
    for (int i = 0; i <= Nx_; i++) {
        ASSERT_ABORT(fscanf(fp_in, "%lf", &xf[i]) == 1);
    }

    ASSERT_ABORT(fscanf(fp_in, "%*s %d", &Ny_) == 1);
    double *const yf = calloc(Ny_+1, sizeof(double));
    CHECK_ALLOC_SUCCESS(yf);
    for (int j = 0; j <= Ny_; j++) {
        ASSERT_ABORT(fscanf(fp_in, "%lf", &yf[j]) == 1);
    }

    /* Read Re, dt, numtstep */
    ASSERT_ABORT(fscanf(fp_in, "%*s %lf", &Re_) == 1);
    ASSERT_ABORT(fscanf(fp_in, "%*s %lf %*s %d", &dt_, &numtstep_) == 2);

    fclose(fp_in);

    /* Make const */
    const int Nx = Nx_, Ny = Ny_;
    const double Re = Re_, dt = dt_;
    const int numtstep = numtstep_;

    /*===== Define variables =================================================*/
    /* Local grid of current process */
    int ilower[2] = {myid * Nx / num_procs + 1, 1};
    int iupper[2] = {(myid+1) * Nx / num_procs, Ny};
    const int Nx_local = iupper[0] - ilower[0] + 1;

    /* Grid variables */
    double *const xc = calloc(Nx_local+2, sizeof(double));
    double *const dx = calloc(Nx_local+2, sizeof(double));
    double *const yc = calloc(Ny+2, sizeof(double));
    double *const dy = calloc(Ny+2, sizeof(double));
    CHECK_ALLOC_SUCCESS(xc, dx, yc, dy);

    /* Pressure and velocities */
    double (*const p      )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const p_next )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const p_prime)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    CHECK_ALLOC_SUCCESS(p, p_next, p_prime);

    double (*const u1      )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const u1_next )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const u1_star )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const u1_tilde)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    CHECK_ALLOC_SUCCESS(u1, u1_next, u1_star, u1_tilde);

    double (*const u2      )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const u2_next )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const u2_star )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const u2_tilde)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    CHECK_ALLOC_SUCCESS(u2, u2_next, u2_star, u2_tilde);

    double (*const U1     )[Ny+2] = calloc(Nx_local+1, sizeof(double [Ny+2]));
    double (*const U1_next)[Ny+2] = calloc(Nx_local+1, sizeof(double [Ny+2]));
    double (*const U1_star)[Ny+2] = calloc(Nx_local+1, sizeof(double [Ny+2]));
    CHECK_ALLOC_SUCCESS(U1, U1_next, U1_star);

    double (*const U2     )[Ny+1] = calloc(Nx_local+2, sizeof(double [Ny+1]));
    double (*const U2_next)[Ny+1] = calloc(Nx_local+2, sizeof(double [Ny+1]));
    double (*const U2_star)[Ny+1] = calloc(Nx_local+2, sizeof(double [Ny+1]));
    CHECK_ALLOC_SUCCESS(U2, U2_next, U2_star);

    /* Auxilary variables */
    double (*const N1     )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const N1_prev)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const N2     )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const N2_prev)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    CHECK_ALLOC_SUCCESS(N1, N1_prev, N2, N2_prev);

    /* For TDMA */
    double (*const C1  )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const C2  )[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const RHS1)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    double (*const RHS2)[Ny+2] = calloc(Nx_local+2, sizeof(double [Ny+2]));
    CHECK_ALLOC_SUCCESS(C1, C2, RHS1, RHS2);

    double *const kx_W = calloc(Nx_local+2, sizeof(double));
    double *const kx_P = calloc(Nx_local+2, sizeof(double));
    double *const kx_E = calloc(Nx_local+2, sizeof(double));
    double *const ky_S = calloc(Ny+2, sizeof(double));
    double *const ky_P = calloc(Ny+2, sizeof(double));
    double *const ky_N = calloc(Ny+2, sizeof(double));
    CHECK_ALLOC_SUCCESS(kx_W, kx_P, kx_E, ky_S, ky_P, ky_N);

    double (*const a_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    double (*const b_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    double (*const c_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    double (*const d_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    double (*const e_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    double (*const x_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    double (*const y_C)[Nx_local+2] = calloc(Ny+2, sizeof(double [Nx_local+2]));
    CHECK_ALLOC_SUCCESS(a_C, b_C, c_C, d_C, e_C, x_C, y_C);

    double *const a_u = calloc(Ny+2, sizeof(double));
    double *const b_u = calloc(Ny+2, sizeof(double));
    double *const c_u = calloc(Ny+2, sizeof(double));
    double *const d_u = calloc(Ny+2, sizeof(double));
    double *const e_u = calloc(Ny+2, sizeof(double));
    double *const x_u = calloc(Ny+2, sizeof(double));
    double *const y_u = calloc(Ny+2, sizeof(double));
    CHECK_ALLOC_SUCCESS(a_u, b_u, c_u, d_u, e_u, x_u, y_u);

    /* For HYPRE */
    HYPRE_StructGrid grid;
    HYPRE_StructStencil stencil;
    HYPRE_StructMatrix matrix;
    HYPRE_StructVector rhsvec;
    HYPRE_StructVector resvec;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond;
    double zeros[Nx_local*Ny];
    double final_res_norm;
    int final_num_iter;

    /*===== Calculate grid variables =========================================*/
    for (int i = 1; i <= Nx_local; i++) {
        dx[i] = xf[i + ilower[0]-1] - xf[i-1 + ilower[0]-1];
        xc[i] = (xf[i + ilower[0]-1] + xf[i-1 + ilower[0]-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        dy[j] = yf[j] - yf[j-1];
        yc[j] = (yf[j] + yf[j-1]) / 2;
    }

    /* Ghost cells */
    dx[0] = (myid == 0 ? dx[1] : xf[ilower[0]-1] - xf[ilower[0]-2]);
    dx[Nx_local+1] = (
        myid == num_procs-1
        ? dx[Nx_local]
        : xf[Nx_local+ilower[0]] - xf[Nx_local+ilower[0]-1]);
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];
    xc[0] = (myid == 0 ? 2*xf[0] - xc[1] : (xf[ilower[0]-1] + xf[ilower[0]-2]) / 2);
    xc[Nx_local+1] = (
        myid == num_procs-1
        ? 2*xf[Nx] - xc[Nx_local]
        : (xf[Nx_local+ilower[0]] + xf[Nx_local+ilower[0]-1]) / 2
    );
    yc[0] = 2*yf[0] - yc[1];
    yc[Ny+1] = 2*yf[Ny] - yc[Ny];

    /* Calculate tdma coefficients */
    for (int i = 1; i <= Nx_local; i++) {
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
            int stencil_indices[5] = {0, 1, 2, 3, 4};
            /* Nx_local*Ny grid points, each with 5 stencil points */
            double values[5*Nx_local*Ny];

            int m = 0;
            for (int j = 1; j <= Ny; j++) {
                for (int i = 1; i <= Nx_local; i++) {
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
            const int Nm = Nx_local > Ny ? Nx_local : Ny;
            double values[2*Nm];

            /* Upper boundary */
            {
                int ilowerb[2] = {ilower[0], Ny}, iupperb[2] = {iupper[0], Ny};
                int stencil_indices[2] = {0, 1};

                int m = 0;
                for (int i = 1; i <= Nx_local; i++) {
                    values[2*m] = kx_W[i] + kx_E[i] + ky_S[Ny];
                    values[2*m+1] = 0;
                    m++;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilowerb, iupperb, 2, stencil_indices, values);
            }
            /* Lower boundary */
            {
                int ilowerb[2] = {ilower[0], 1}, iupperb[2] = {iupper[0], 1};
                int stencil_indices[2] = {0, 3};

                int m = 0;
                for (int i = 1; i <= Nx_local; i++) {
                    values[2*m] = kx_W[i] + kx_E[i] + ky_N[1];
                    values[2*m+1] = 0;
                    m++;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilowerb, iupperb, 2, stencil_indices, values);
            }
            /* Left boundary: only for the first process */
            if (myid == 0) {
                int ilowerb[2] = {1, 1}, iupperb[2] = {1, Ny};
                int stencil_indices[2] = {0, 4};

                int m = 0;
                for (int j = 1; j <= Ny; j++) {
                    values[2*m] = kx_E[1] + ky_S[j] + ky_N[j];
                    values[2*m+1] = 0;
                    m++;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilowerb, iupperb, 2, stencil_indices, values);

                /* Upper left corner */
                stencil_indices[0] = 0;
                values[0] = kx_E[1] + ky_S[Ny];
                HYPRE_StructMatrixSetBoxValues(matrix, iupperb, iupperb, 1, stencil_indices, values);

                /* Lower left corner */
                {
                    int stencil_indices[5] = {0, 1, 2, 3, 4};
                    double values[5] = {1, 0, 0, 0, 0};
                    HYPRE_StructMatrixSetBoxValues(matrix, ilowerb, ilowerb, 5, stencil_indices, values);
                }
            }
            /* Right boundary: only for the last process */
            if (myid == num_procs-1) {
                int ilowerb[2] = {Nx, 1}, iupperb[2] = {Nx, Ny};
                int stencil_indices[2] = {0, 2};

                int m = 0;
                for (int j = 1; j <= Ny; j++) {
                    values[2*m] = kx_W[Nx_local] + ky_S[j] + ky_N[j];
                    values[2*m+1] = 0;
                    m++;
                }

                HYPRE_StructMatrixSetBoxValues(matrix, ilowerb, iupperb, 2, stencil_indices, values);

                /* Upper right corner */
                stencil_indices[0] = 0;
                values[0] = kx_W[Nx_local] + ky_S[Ny];
                HYPRE_StructMatrixSetBoxValues(matrix, iupperb, iupperb, 1, stencil_indices, values);

                /* Lower right corner */
                stencil_indices[0] = 0;
                values[0] = kx_W[Nx_local] + ky_N[1];
                HYPRE_StructMatrixSetBoxValues(matrix, ilowerb, ilowerb, 1, stencil_indices, values);
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
        HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_StructPCGSetMaxIter(solver, 1000);
        HYPRE_StructPCGSetTol(solver, 1.0e-06);
        HYPRE_StructPCGSetTwoNorm(solver, 1);
        HYPRE_StructPCGSetRelChange(solver, 0);
        HYPRE_StructPCGSetPrintLevel(solver, 1); /* print each CG iteration */
        HYPRE_StructPCGSetLogging(solver, 1);

        /* Use symmetric SMG as preconditioner */
        HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_StructSMGSetMemoryUse(precond, 0);
        HYPRE_StructSMGSetMaxIter(precond, 10);
        HYPRE_StructSMGSetTol(precond, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond);
        HYPRE_StructSMGSetNumPreRelax(precond, 3);
        HYPRE_StructSMGSetNumPostRelax(precond, 3);

        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);

        HYPRE_StructPCGSetup(solver, matrix, rhsvec, resvec);
    }
    /* Fill zeros array */
    for (int i = 0; i < Nx_local*Ny; i++)
        zeros[i] = 0;

    /*===== Initialize flow ==================================================*/
    for (int i = 0; i <= Nx_local; i++) {
        U1[i][Ny+1] = 2;
    }
    for (int i = 1; i <= Nx_local; i++) {
        u1[i][Ny+1] = 2;
    }

    /*===== Tic ==============================================================*/
    if (myid == 0) {
        clock_gettime(CLOCK_REALTIME, &start_time_total);
    }

    /*===== Main loop ========================================================*/
    for (int tstep = 1; tstep <= numtstep; tstep++) {
        /* Exchange p, u1 and u2 between the adjacent processes */
        if (myid != num_procs-1) {
            MPI_Send(p[Nx_local], Ny+2, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
            MPI_Send(u1[Nx_local], Ny+2, MPI_DOUBLE, myid+1, 1, MPI_COMM_WORLD);
            MPI_Send(u2[Nx_local], Ny+2, MPI_DOUBLE, myid+1, 2, MPI_COMM_WORLD);
        }
        if (myid != 0) {
            MPI_Recv(p[0], Ny+2, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u1[0], Ny+2, MPI_DOUBLE, myid-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2[0], Ny+2, MPI_DOUBLE, myid-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(p[1], Ny+2, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
            MPI_Send(u1[1], Ny+2, MPI_DOUBLE, myid-1, 1, MPI_COMM_WORLD);
            MPI_Send(u2[1], Ny+2, MPI_DOUBLE, myid-1, 2, MPI_COMM_WORLD);
        }
        if (myid != num_procs-1) {
            MPI_Recv(p[Nx_local+1], Ny+2, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u1[Nx_local+1], Ny+2, MPI_DOUBLE, myid+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2[Nx_local+1], Ny+2, MPI_DOUBLE, myid+1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Calculate N */
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u1_e = (u1[i][j] * dx[i+1] + u1[i+1][j] * dx[i]) / (dx[i] + dx[i+1]);
                double u2_e = (u2[i][j] * dx[i+1] + u2[i+1][j] * dx[i]) / (dx[i] + dx[i+1]);
                double u1_w = (u1[i-1][j] * dx[i] + u1[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
                double u2_w = (u2[i-1][j] * dx[i] + u2[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
                double u1_n = (u1[i][j] * dy[j+1] + u1[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
                double u2_n = (u2[i][j] * dy[j+1] + u2[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
                double u1_s = (u1[i][j-1] * dy[j] + u1[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);
                double u2_s = (u2[i][j-1] * dy[j] + u2[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);

                N1[i][j] = (U1[i][j]*u1_e - U1[i-1][j]*u1_w) / dx[i]
                    + (U2[i][j]*u1_n - U2[i][j-1]*u1_s) / dy[j];
                N2[i][j] = (U1[i][j]*u2_e - U1[i-1][j]*u2_w) / dx[i]
                    + (U2[i][j]*u2_n - U2[i][j-1]*u2_s) / dy[j];
            }
        }

        /* Calculate RHS */
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny; j++) {
                RHS1[i][j] = -dt/2 * (3*N1[i][j] - N1_prev[i][j])
                    - dt * (p[i+1][j] - p[i-1][j]) / (xc[i+1] - xc[i-1])
                    + 2 * (kx_W[i]*u1[i-1][j] + kx_E[i]*u1[i+1][j]
                           + ky_S[j]*u1[i][j-1] + ky_N[j]*u1[i][j+1]
                           - (kx_W[i]+kx_E[i]+ky_S[j]+ky_N[j])*u1[i][j]);
                RHS2[i][j] = -dt/2 * (3*N2[i][j] - N2_prev[i][j])
                    - dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1])
                    + 2 * (kx_W[i]*u2[i-1][j] + kx_E[i]*u2[i+1][j]
                           + ky_S[j]*u2[i][j-1] + ky_N[j]*u2[i][j+1]
                           - (kx_W[i]+kx_E[i]+ky_S[j]+ky_N[j])*u2[i][j]);
            }
        }

        /* Calcuate C */
        {
            for (int j = 1; j <= Ny; j++) {
                /* Initialize TDMA matrix */
                for (int i = 1; i <= Nx_local; i++) {
                    a_C[j][i] = -kx_W[i];
                    b_C[j][i] = kx_P[i];
                    c_C[j][i] = -kx_E[i];
                    d_C[j][i] = RHS1[i][j];
                    e_C[j][i] = RHS2[i][j];
                }
                if (myid == 0) {
                    b_C[j][1] = kx_W[1] + kx_P[1];
                }
                if (myid == num_procs-1) {
                    b_C[j][Nx_local] = kx_P[Nx_local] + kx_E[Nx_local];
                }

                /* Forward elimination */
                if (myid == 0) {
                    forward(a_C[j], b_C[j], c_C[j], d_C[j], e_C[j], 1, Nx_local);
                }
                else {
                    double firsts[4];
                    MPI_Recv(firsts, 4, MPI_DOUBLE, myid-1, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    b_C[j][0] = firsts[0];
                    c_C[j][0] = firsts[1];
                    d_C[j][0] = firsts[2];
                    e_C[j][0] = firsts[3];
                    forward(a_C[j], b_C[j], c_C[j], d_C[j], e_C[j], 0, Nx_local);
                }
                if (myid != num_procs-1) {
                    double lasts[4] = {b_C[j][Nx_local], c_C[j][Nx_local], d_C[j][Nx_local], e_C[j][Nx_local]};
                    MPI_Send(lasts, 4, MPI_DOUBLE, myid+1, j, MPI_COMM_WORLD);
                }
            }
            for (int j = 1; j <= Ny; j++) {
                /* Back substitution */
                if (myid == num_procs-1) {
                    x_C[j][Nx_local] = d_C[j][Nx_local] / b_C[j][Nx_local];
                    y_C[j][Nx_local] = e_C[j][Nx_local] / b_C[j][Nx_local];
                    back(b_C[j], c_C[j], d_C[j], e_C[j], x_C[j], y_C[j], 1, Nx_local);
                }
                else {
                    double lasts[2];
                    MPI_Recv(lasts, 2, MPI_DOUBLE, myid+1, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    x_C[j][Nx_local+1] = lasts[0];
                    y_C[j][Nx_local+1] = lasts[1];
                    back(b_C[j], c_C[j], d_C[j], e_C[j], x_C[j], y_C[j], 1, Nx_local+1);
                }
                if (myid != 0) {
                    double firsts[2] = {x_C[j][1], y_C[j][1]};
                    MPI_Send(firsts, 2, MPI_DOUBLE, myid-1, j, MPI_COMM_WORLD);
                }
                for (int i = 1; i <= Nx_local; i++) {
                    C1[i][j] = x_C[j][i];
                    C2[i][j] = y_C[j][i];
                }
            }
        }

        /* Calculate u_star */
        {
            for (int i = 1; i <= Nx_local; i++) {
                for (int j = 1; j <= Ny; j++) {
                    a_u[j] = -ky_S[j];
                    b_u[j] = ky_P[j];
                    c_u[j] = -ky_N[j];
                    d_u[j] = C1[i][j];
                    e_u[j] = C2[i][j];
                }
                b_u[1] = ky_S[1] + ky_P[1];
                b_u[Ny] = ky_P[Ny] + ky_N[Ny];

                forward(a_u, b_u, c_u, d_u, e_u, 1, Ny);
                x_u[Ny] = d_u[Ny] / b_u[Ny];
                y_u[Ny] = e_u[Ny] / b_u[Ny];
                back(b_u, c_u, d_u, e_u, x_u, y_u, 1, Ny);

                for (int j = 1; j <= Ny; j++) {
                    u1_star[i][j] = x_u[j] + u1[i][j];
                    u2_star[i][j] = y_u[j] + u2[i][j];
                }
            }
        }

        /* Calculate u_tilde */
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_tilde[i][j] = u1_star[i][j] + dt * (p[i+1][j] - p[i-1][j]) / (xc[i+1] - xc[i-1]);
                u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        /* Exchange u1_tilde between the adjacent processes */
        if (myid != num_procs-1) {
            MPI_Send(u1_tilde[Nx_local], Ny+2, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
        }
        if (myid != 0) {
            MPI_Recv(u1_tilde[0], Ny+2, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(u1_tilde[1], Ny+2, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
        }
        if (myid != num_procs-1) {
            MPI_Recv(u1_tilde[Nx_local+1], Ny+2, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Calculate U_star */
        int istart = myid == 0? 1 : 0;
        int iend = myid == num_procs-1? Nx_local-1 : Nx_local;
        for (int i = istart; i <= iend; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_star[i][j] = (u1_tilde[i][j]*dx[i+1] + u1_tilde[i+1][j]*dx[i]) / (dx[i] + dx[i+1])
                    - dt * (p[i+1][j] - p[i][j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                U2_star[i][j] = (u2_tilde[i][j]*dy[j+1] + u2_tilde[i][j+1]*dy[j]) / (dy[j] + dy[j+1])
                    - dt * (p[i][j+1] - p[i][j]) / (yc[j+1] - yc[j]);
            }
        }

        /* Calculate p_prime */
        {
            int m;
            double values[Nx_local*Ny];

            m = 0;
            if (myid == 0) {
                for (int j = 1; j <= Ny; j++) {
                    for (int i = 1; i <= Nx_local; i++) {
                        double Q = 1 / (2.*Re) * ((U1_star[i][j] - U1_star[i-1][j]) / dx[i]
                                                  + (U2_star[i][j] - U2_star[i][j-1]) / dy[j]);
                        values[m] = (i == 1 && j == 1) ? 0 : -Q;
                        m++;
                    }
                }
            }
            else {
                for (int j = 1; j <= Ny; j++) {
                    for (int i = 1; i <= Nx_local; i++) {
                        values[m] = -1 / (2.*Re) * ((U1_star[i][j] - U1_star[i-1][j]) / dx[i]
                                                    + (U2_star[i][j] - U2_star[i][j-1]) / dy[j]);
                        m++;
                    }
                }
            }
            HYPRE_StructVectorSetBoxValues(rhsvec, ilower, iupper, values);
            HYPRE_StructVectorSetBoxValues(resvec, ilower, iupper, zeros);

            HYPRE_StructVectorAssemble(rhsvec);
            HYPRE_StructVectorAssemble(resvec);

            HYPRE_StructPCGSolve(solver, matrix, rhsvec, resvec);

            HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
            if (final_res_norm >= 1e-3 && myid == 0) {
                printf("warning: not converged!\n");
            }
            else if (final_res_norm >= 1e3 && myid == 0) {
                printf("error: diverged!\n");
                break;
            }
            HYPRE_StructPCGGetNumIterations(solver, &final_num_iter);
            if (final_num_iter > 10 && myid == 0) {
                printf("num_iter > 10!\n");
            }

            HYPRE_StructVectorGetBoxValues(resvec, ilower, iupper, values);
            m = 0;
            for (int j = 1; j <= Ny; j++) {
                for (int i = 1; i <= Nx_local; i++) {
                    p_prime[i][j] = values[m];
                    m++;
                }
            }
        }

        /* Calculate p_next */
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny; j++) {
                p_next[i][j] = p[i][j] + p_prime[i][j];
            }
        }

        /* Exchange p_prime between the adjacent processes */
        if (myid != num_procs-1) {
            MPI_Send(p_prime[Nx_local], Ny+2, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
        }
        if (myid != 0) {
            MPI_Recv(p_prime[0], Ny+2, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(p_prime[1], Ny+2, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
        }
        if (myid != num_procs-1) {
            MPI_Recv(p_prime[Nx_local+1], Ny+2, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Calculate u_next */
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[i][j] = u1_star[i][j] - dt * (p_prime[i+1][j] - p_prime[i-1][j]) / (xc[i+1] - xc[i-1]);
                u2_next[i][j] = u2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j-1]) / (yc[j+1] - yc[j-1]);
            }
        }

        /* Calculate U_next */
        for (int i = istart; i <= iend; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_next[i][j] = U1_star[i][j] - dt * (p_prime[i+1][j] - p_prime[i][j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx_local; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                U2_next[i][j] = U2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j]) / (yc[j+1] - yc[j]);
            }
        }

        /* Set velocity boundary conditions */
        /* Upper boundary */
        for (int i = 1; i <= Nx_local; i++) {
            u1_next[i][Ny+1] = 2 - u1_next[i][Ny];
            u2_next[i][Ny+1] = -u2_next[i][Ny];
        }
        for (int i = 0; i <= Nx_local; i++) {
            U1_next[i][Ny+1] = 2 - U1_next[i][Ny];
        }
        /* Lower boundary */
        for (int i = 1; i <= Nx_local; i++) {
            u1_next[i][0] = -u1_next[i][1];
            u2_next[i][0] = -u2_next[i][1];
        }
        for (int i = 0; i <= Nx_local; i++) {
            U1_next[i][0] = -U1_next[i][1];
        }
        /* Left boundary; only for the first process */
        if (myid == 0) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[0][j] = -u1_next[1][j];
                u2_next[0][j] = -u2_next[1][j];
            }
            for (int j = 0; j <= Ny; j++) {
                U2_next[0][j] = -U2_next[1][j];
            }
        }
        /* Right boudnary; only for the last process */
        if (myid == num_procs-1) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[Nx_local+1][j] = -u1_next[Nx_local][j];
                u2_next[Nx_local+1][j] = -u2_next[Nx_local][j];
            }
            for (int j = 0; j <= Ny; j++) {
                U2_next[Nx_local+1][j] = -U2_next[Nx_local][j];
            }
        }

        /* Update for next time step */
        memcpy(p, p_next, sizeof(double)*(Nx_local+2)*(Ny+2));
        memcpy(u1, u1_next, sizeof(double)*(Nx_local+2)*(Ny+2));
        memcpy(u2, u2_next, sizeof(double)*(Nx_local+2)*(Ny+2));
        memcpy(U1, U1_next, sizeof(double)*(Nx_local+1)*(Ny+2));
        memcpy(U2, U2_next, sizeof(double)*(Nx_local+2)*(Ny+1));
        memcpy(N1_prev, N1, sizeof(double)*(Nx_local+2)*(Ny+2));
        memcpy(N2_prev, N2, sizeof(double)*(Nx_local+2)*(Ny+2));

        if (tstep % 100 == 0 && myid == 0) {
            printf("tstep: %d\n", tstep);
        }
    }

    /*===== Toc ==============================================================*/
    if (myid == 0) {
        clock_gettime(CLOCK_REALTIME, &end_time_total);
        double elapsed_time = (end_time_total.tv_sec-start_time_total.tv_sec) + (end_time_total.tv_nsec-start_time_total.tv_nsec)/1.e9;
        printf("total elapsed time (s): %.6lf\n", elapsed_time);
    }

    /*===== Export result ====================================================*/
    if (myid == 0) {
        double U1_total[Nx+1][Ny+2];

        /* Collect velocity field of all processes */
        memcpy(&U1_total[0][0], &U1[0][0], sizeof(double)*(Nx_local+1)*(Ny+2));
        for (int p = 1; p < num_procs; p++) {
            int a = p*Nx/num_procs + 1, b = (p+1)*Nx/num_procs;
            MPI_Recv(&U1_total[a-1][0], (b-a+2)*(Ny+2), MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Calculate streamfunction */
        double psi[Nx+1][Ny+1];
        for (int i = 0; i <= Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                psi[i][j] = 0;
            }
        }

        for (int i = 1; i <= Nx-1; i++) {
            for (int j = 1; j <= Ny-1; j++) {
                psi[i][j] = psi[i][j-1] + dy[j] * U1_total[i][j];
            }
        }

        /* Write to output file */
        FILE *fp_out = fopen("result/cavity_result.txt", "w");
        for (int i = 0; i <= Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                fprintf(fp_out, "%17.14lf ", psi[i][j]);
            }
            fprintf(fp_out, "\n");
        }
        fclose(fp_out);
    }
    else {
        MPI_Send(&U1[0][0], (Nx_local+1)*(Ny+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    /*===== Finalize program =================================================*/
    /* Free memory */
    free(xf); free(yf);
    free(xc); free(dx); free(yc); free(dy);

    free(p); free(p_next); free(p_prime);
    free(u1); free(u1_next); free(u1_star); free(u1_tilde);
    free(u2); free(u2_next); free(u2_star); free(u2_tilde);
    free(U1); free(U1_next); free(U1_star);
    free(U2); free(U2_next); free(U2_star);

    free(N1); free(N1_prev); free(N2); free(N2_prev);

    free(C1); free(C2); free(RHS1); free(RHS2);

    free(kx_W); free(kx_P); free(kx_E);
    free(ky_S); free(ky_P); free(ky_N);

    free(a_C); free(b_C); free(c_C); free(d_C); free(e_C); free(x_C); free(y_C);
    free(a_u); free(b_u); free(c_u); free(d_u); free(e_u); free(x_u); free(y_u);

    HYPRE_StructGridDestroy(grid);
    HYPRE_StructStencilDestroy(stencil);
    HYPRE_StructMatrixDestroy(matrix);
    HYPRE_StructVectorDestroy(rhsvec);
    HYPRE_StructVectorDestroy(resvec);
    HYPRE_StructPCGDestroy(solver);

    /* Finalize Hypre */
    HYPRE_Finalize();

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}
