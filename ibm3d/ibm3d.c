#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include "geo3d.h"

/* Define and allocate 3-D array ARRNAME whose size is NX * NY * NZ. */
#define ALLOC3D(arrname, nx, ny, nz) \
    double (*arrname)[ny][nz] = calloc(nx, sizeof(double [ny][nz]))

#define FOR_ALL_CELL(i, j, k) \
    for (int i = 1; i <= Nx; i++) \
        for (int j = 1; j <= Ny; j++) \
            for (int k = 1; k <= Nz; k++)
#define FOR_ALL_XSTAG(i, j, k) \
    for (int i = 0; i <= Nx; i++) \
        for (int j = 0; j <= Ny+1; j++) \
            for (int k = 0; k <= Nz+1; k++)
#define FOR_ALL_YSTAG(i, j, k) \
    for (int i = 0; i <= Nx+1; i++) \
        for (int j = 0; j <= Ny; j++) \
            for (int k = 0; k <= Nz+1; k++)
#define FOR_ALL_ZSTAG(i, j, k) \
    for (int i = 0; i <= Nx+1; i++) \
        for (int j = 0; j <= Ny+1; j++) \
            for (int k = 0; k <= Nz; k++)

/* Index of cell (i, j, k), ranging from 1 to Nx * Ny * Nz. */
#define IDXFLAT(i, j, k) (Ny*Nz*(i-1) + Nz*(j-1) + (k))
#define SWAP(a, b) do {typeof(a) tmp = a; a = b; b = tmp;} while (0)

/* Index of adjacent cells in 3-d cartesian coordinate */
static const int adj[6][3] = {
    {0, 1, 0}, {1, 0, 0}, {0, -1, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, 1}
};
/* Order is:
            5
          z |  0
            | / y
            |/
 3 ---------+--------- 1
           /|        x
          / |
         2  |
            4
*/


static inline HYPRE_IJMatrix create_matrix(
    const int Nx, const int Ny, const int Nz,
    const double xc[const restrict static Nx+2],
    const double yc[const restrict static Ny+2],
    const double zc[const restrict static Nz+2],
    const double lvset[const restrict static Nx+2][Ny+2][Nz+2],
    const int flag[const restrict static Nx+2][Ny+2][Nz+2],
    const double kx_W[const restrict static Nx+2],
    const double kx_E[const restrict static Nx+2],
    const double ky_S[const restrict static Ny+2],
    const double ky_N[const restrict static Ny+2],
    const double kz_D[const restrict static Nz+2],
    const double kz_U[const restrict static Nz+2],
    const int type
);

static void get_interp_info(
    const int Nx, const int Ny, const int Nz,
    const double xc[const restrict static Nx+2],
    const double yc[const restrict static Ny+2],
    const double zc[const restrict static Nz+2],
    const double lvset[const restrict static Nx+2][Ny+2][Nz+2],
    const int i, const int j, const int k,
    int interp_idx[const restrict static 8],
    double interp_coeff[const restrict static 8]
);

static FILE *fopen_check(
    const char *restrict filename, const char *restrict modes
);

void calc_flux(
    const int Nx, const int Ny, const int Nz,
    const double dx[const restrict static Nx+2],
    const double dy[const restrict static Ny+2],
    const double dz[const restrict static Nz+2],
    const double u1[const restrict static Nx+2][Ny+2][Nz+2],
    const double u2[const restrict static Nx+2][Ny+2][Nz+2],
    const double u3[const restrict static Nx+2][Ny+2][Nz+2],
    const double U1[const restrict static Nx+1][Ny+2][Nz+2],
    const double U2[const restrict static Nx+2][Ny+1][Nz+2],
    const double U3[const restrict static Nx+2][Ny+2][Nz+1],
    double N1[const restrict static Nx+2][Ny+2][Nz+2],
    double N2[const restrict static Nx+2][Ny+2][Nz+2],
    double N3[const restrict static Nx+2][Ny+2][Nz+2]
);

int main(int argc, char **argv) {
    /****** Initialize program and parse arguments. ***************************/
    /*===== Initialize MPI. ==================================================*/

    /* Id of current process. */
    int myid;
    /* Number of all processes. */
    int num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /*===== Initialize HYPRE. ================================================*/

    HYPRE_Init();

    /* For the present, the number of process must be 1. */
    if (num_procs != 1) {
        if (myid == 0)
            printf("Must run with 1 processors!\n");
        MPI_Finalize();
        return 0;
    }

    /****** Read input file. **************************************************/
    /*===== Define input parameters. =========================================*/

    /* Files. */
    FILE *fp_in, *fp_poly;

    /* Name of stl file containing polyhedron info. */
    char stl_file[100];
    /* Polyhedron read from stl file. */
    Polyhedron *poly;

    /* Number of cells in x, y, and z direction, respectively. */
    int Nx, Ny, Nz;
    /* Coordinates of cell faces. */
    double *xf, *yf, *zf;
    /* Reynolds number. */
    double Re;
    /* Delta t. */
    double dt;
    /* Totla number of time steps. */
    int numtstep;

    /* Initialize velocities and pressure from file? (T/F) */
    int init_using_file;
    /* Names of velocity input files for initialization. */
    char init_file_u1[100], init_file_u2[100], init_file_u3[100];
    /* Names of pressure input file for initialization. */
    char init_file_p[100];

    /* Name of velocity output files for result export */
    char output_file_u1[100], output_file_u2[100], output_file_u3[100];
    /* Name of pressure output file for result export */
    char output_file_p[100];

    /*===== Read inputs ======================================================*/

    /* Open input file */
    printf("Read input file\n");
    fp_in = fopen_check("ibm3d.in", "r");

    /* Read stl file */
    fscanf(fp_in, "%*s %s", stl_file);
    printf("Read polyhedron file: %s\n", stl_file);
    fp_poly = fopen_check(stl_file, "rb");

    poly = Polyhedron_new();
    Polyhedron_read_stl(poly, fp_poly);

    /* Read grid geometry */
    fscanf(fp_in, "%*s %d", &Nx);
    xf = calloc(Nx+1, sizeof(double));
    for (int i = 0; i <= Nx; i++) {
        fscanf(fp_in, "%lf", &xf[i]);
    }

    fscanf(fp_in, "%*s %d", &Ny);
    yf = calloc(Ny+1, sizeof(double));
    for (int j = 0; j <= Ny; j++) {
        fscanf(fp_in, "%lf", &yf[j]);
    }

    fscanf(fp_in, "%*s %d", &Nz);
    zf = calloc(Nz+1, sizeof(double));
    for (int k = 0; k <= Nz; k++) {
        fscanf(fp_in, "%lf", &zf[k]);
    }

    /* Read Reynolds number, delta t, and number of time steps */
    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep);

    /* Read initialization file names */
    fscanf(fp_in, "%*s %d", &init_using_file);
    fscanf(fp_in, "%*s %s", init_file_u1);
    fscanf(fp_in, "%*s %s", init_file_u2);
    fscanf(fp_in, "%*s %s", init_file_u3);
    fscanf(fp_in, "%*s %s", init_file_p);

    /* Read output file names */
    fscanf(fp_in, "%*s %s", output_file_u1);
    fscanf(fp_in, "%*s %s", output_file_u2);
    fscanf(fp_in, "%*s %s", output_file_u3);
    fscanf(fp_in, "%*s %s", output_file_p);

    fclose(fp_in);
    fclose(fp_poly);

    /****** Print input statistics ********************************************/

    /* Mesh statistics */
    printf("\n");
    printf("Input mesh size: %d x %d x %d\n", Nx, Ny, Nz);
    printf("  xmin: %10.4lf, xmax: %10.4lf\n", xf[0], xf[Nx]);
    printf("  ymin: %10.4lf, ymax: %10.4lf\n", yf[0], yf[Ny]);
    printf("  zmin: %10.4lf, zmax: %10.4lf\n", zf[0], zf[Ny]);

    /* Polyhedron statistics */
    Polyhedron_print_stats(poly);

    /* Reynolds number and delta t */
    printf("\n");
    printf("Reynolds no.: %.6lf\n", Re);
    printf("delta t     : %.6lf\n", dt);

    /****** Define variables **************************************************/
    /*===== Grid variables ===================================================*/

    /* Cell widths */
    double dx[Nx+2], dy[Ny+2], dz[Nz+2];
    /* Cell centroid coordinates */
    double xc[Nx+2], yc[Ny+2], zc[Nz+2];

    /* Level set function (signed distance function) at cell centroids */
    double (*const lvset)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    /* Flag of each cell (1: fluid cell, 2: ghost cell, 0: solid cell) */
    int (*const flag)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));

    /*===== Velocities and pressure. =========================================*/

    ALLOC3D(      u1      , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      u1_next , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u1_star , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u1_tilde, Nx+2, Ny+2, Nz+2);

    ALLOC3D(      u2      , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      u2_next , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u2_star , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u2_tilde, Nx+2, Ny+2, Nz+2);

    ALLOC3D(      u3      , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      u3_next , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u3_star , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u3_tilde, Nx+2, Ny+2, Nz+2);

    ALLOC3D(      U1      , Nx+1, Ny+2, Nz+2);
    ALLOC3D(      U1_next , Nx+1, Ny+2, Nz+2);
    ALLOC3D(const U1_star , Nx+1, Ny+2, Nz+2);

    ALLOC3D(      U2      , Nx+2, Ny+1, Nz+2);
    ALLOC3D(      U2_next , Nx+2, Ny+1, Nz+2);
    ALLOC3D(const U2_star , Nx+2, Ny+1, Nz+2);

    ALLOC3D(      U3      , Nx+2, Ny+2, Nz+1);
    ALLOC3D(      U3_next , Nx+2, Ny+2, Nz+1);
    ALLOC3D(const U3_star , Nx+2, Ny+2, Nz+1);

    ALLOC3D(      p       , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      p_next  , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const p_prime , Nx+2, Ny+2, Nz+2);

    /*===== Fluxes. ==========================================================*/

    ALLOC3D(N1     , Nx+2, Ny+2, Nz+2);
    ALLOC3D(N1_prev, Nx+2, Ny+2, Nz+2);
    ALLOC3D(N2     , Nx+2, Ny+2, Nz+2);
    ALLOC3D(N2_prev, Nx+2, Ny+2, Nz+2);
    ALLOC3D(N3     , Nx+2, Ny+2, Nz+2);
    ALLOC3D(N3_prev, Nx+2, Ny+2, Nz+2);

    /*===== Derivative coefficients. =========================================*/

    double kx_W[Nx+2], kx_E[Nx+2];
    double ky_S[Ny+2], ky_N[Ny+2];
    double kz_D[Nz+2], kz_U[Nz+2];

    /*===== HYPRE matrices, vectors, solvers, and arrays. ====================*/

    HYPRE_IJMatrix     A_u1, A_u2, A_u3;
    HYPRE_ParCSRMatrix parcsr_A_u1, parcsr_A_u2, parcsr_A_u3;
    HYPRE_IJVector     b_u1, b_u2, b_u3;
    HYPRE_ParVector    par_b_u1, par_b_u2, par_b_u3;
    HYPRE_IJVector     x_u1, x_u2, x_u3;
    HYPRE_ParVector    par_x_u1, par_x_u2, par_x_u3;

    HYPRE_IJMatrix     A_p;
    HYPRE_ParCSRMatrix parcsr_A_p;
    HYPRE_IJVector     b_p;
    HYPRE_ParVector    par_b_p;
    HYPRE_IJVector     x_p;
    HYPRE_ParVector    par_x_p;

    HYPRE_Solver solver_u1, solver_u2, solver_u3;
    HYPRE_Solver solver_p;
    HYPRE_Solver precond;

    int *vector_rows;
    double *vector_values_u1, *vector_values_u2, *vector_values_u3;
    double *vector_values_p;
    double *vector_zeros, *vector_res;

    int num_iters;
    double final_res_norm;

    /*===== Misc. ============================================================*/

    struct timespec t_start, t_end;
    FILE *fp_out;

    /****** Calculate grid variables ******************************************/

    /* Cell width and centroid coordinates */
    for (int i = 1; i <= Nx; i++) {
        dx[i] = xf[i] - xf[i-1];
        xc[i] = (xf[i] + xf[i-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        dy[j] = yf[j] - yf[j-1];
        yc[j] = (yf[j] + yf[j-1]) / 2;
    }
    for (int k = 1; k <= Nz; k++) {
        dz[k] = zf[k] - zf[k-1];
        zc[k] = (zf[k] + zf[k-1]) / 2;
    }

    /* Ghost cells */
    dx[0] = dx[1];
    dx[Nx+1] = dx[Nx];
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];
    dz[0] = dz[1];
    dz[Nz+1] = dz[Nz];

    xc[0] = 2*xf[0] - xc[1];
    xc[Nx+1] = 2*xf[Nx] - xc[Nx];
    yc[0] = 2*yf[0] - yc[1];
    yc[Ny+1] = 2*yf[Ny] - yc[Ny];
    zc[0] = 2*zf[0] - zc[1];
    zc[Nz+1] = 2*zf[Nz] - zc[Nz];

    /* Calculate second order derivative coefficients */
    for (int i = 1; i <= Nx; i++) {
        kx_W[i] = dt / (2*Re * (xc[i] - xc[i-1])*dx[i]);
        kx_E[i] = dt / (2*Re * (xc[i+1] - xc[i])*dx[i]);
    }
    for (int j = 1; j <= Ny; j++) {
        ky_S[j] = dt / (2*Re * (yc[j] - yc[j-1])*dy[j]);
        ky_N[j] = dt / (2*Re * (yc[j+1] - yc[j])*dy[j]);
    }
    for (int k = 1; k <= Nz; k++) {
        kz_D[k] = dt / (2*Re * (zc[k] - zc[k-1])*dz[k]);
        kz_U[k] = dt / (2*Re * (zc[k+1] - zc[k])*dz[k]);
    }

    /****** Calculate level set function **************************************/

    Polyhedron_cpt(poly, Nx+2, Ny+2, Nz+2, xc, yc, zc, lvset, 5);

    // FILE *fp_lvset = fopen_check("lvset.out", "w");
    // fwrite(lvset, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_lvset);
    // fclose(fp_lvset);

    Polyhedron_destroy(poly);

    /****** Calculate flags ***************************************************/

    /* * Level set function is positive           => fluid cell (flag = 1)
       * Level set function if negative and at
           least one adjacent cell is fluid cell  => ghost cell (flag = 2)
       * Otherwise                                => solid cell (flag = 0) */
    FOR_ALL_CELL (i, j, k) {
        if (lvset[i][j][k] > 0) {
            flag[i][j][k] = 1;
        }
        else {
            bool is_ghost_cell = false;
            for (int l = 0; l < 6; l++) {
                int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                if (
                    1 <= ni && ni <= Nx
                    && 1 <= nj && nj <= Ny
                    && 1 <= nk && nk <= Nz
                ) {
                    is_ghost_cell = is_ghost_cell || (lvset[ni][nj][nk] > 0);
                }
            }
            flag[i][j][k] = is_ghost_cell ? 2 : 0;
        }
    }

    /****** Initialize HYPRE variables. ***************************************/
    /*===== Initialize matrices. =============================================*/

    A_u1 = create_matrix(
        Nx, Ny, Nz, xc, yc, zc,
        lvset, flag,
        kx_W, kx_E, ky_S, ky_N, kz_D, kz_U,
        1
    );
    HYPRE_IJMatrixGetObject(A_u1, (void **)&parcsr_A_u1);

    A_u2 = create_matrix(
        Nx, Ny, Nz, xc, yc, zc,
        lvset, flag,
        kx_W, kx_E, ky_S, ky_N, kz_D, kz_U,
        2
    );
    HYPRE_IJMatrixGetObject(A_u2, (void **)&parcsr_A_u2);

    A_u3 = create_matrix(
        Nx, Ny, Nz, xc, yc, zc,
        lvset, flag,
        kx_W, kx_E, ky_S, ky_N, kz_D, kz_U,
        3
    );
    HYPRE_IJMatrixGetObject(A_u3, (void **)&parcsr_A_u3);

    A_p = create_matrix(
        Nx, Ny, Nz, xc, yc, zc,
        lvset, flag,
        kx_W, kx_E, ky_S, ky_N, kz_D, kz_U,
        4
    );
    HYPRE_IJMatrixGetObject(A_p, (void **)&parcsr_A_p);

    /*===== Initialize vectors. ==============================================*/

    /* RHS vectors. */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &b_u1);
    HYPRE_IJVectorSetObjectType(b_u1, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_u1);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &b_u2);
    HYPRE_IJVectorSetObjectType(b_u2, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_u2);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &b_u3);
    HYPRE_IJVectorSetObjectType(b_u3, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_u3);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &b_p);
    HYPRE_IJVectorSetObjectType(b_p, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_p);

    /* Solution vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &x_u1);
    HYPRE_IJVectorSetObjectType(x_u1, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_u1);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &x_u2);
    HYPRE_IJVectorSetObjectType(x_u2, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_u2);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &x_u3);
    HYPRE_IJVectorSetObjectType(x_u3, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_u3);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &x_p);
    HYPRE_IJVectorSetObjectType(x_p, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_p);

    vector_rows = calloc(Nx*Ny*Nz, sizeof(int));
    vector_values_u1 = calloc(Nx*Ny*Nz, sizeof(double));
    vector_values_u2 = calloc(Nx*Ny*Nz, sizeof(double));
    vector_values_u3 = calloc(Nx*Ny*Nz, sizeof(double));
    vector_values_p = calloc(Nx*Ny*Nz, sizeof(double));
    vector_zeros = calloc(Nx*Ny*Nz, sizeof(double));
    vector_res = calloc(Nx*Ny*Nz, sizeof(double));

    for (int i = 0; i < Nx*Ny*Nz; i++) {
        vector_rows[i] = i+1;
    }

    /*===== Initialize solvers. ==============================================*/

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_u1);
    HYPRE_ParCSRBiCGSTABSetLogging(solver_u1, 1);
    HYPRE_BiCGSTABSetMaxIter(solver_u1, 1000);
    HYPRE_BiCGSTABSetTol(solver_u1, 1e-6);
    // HYPRE_BiCGSTABSetPrintLevel(solver_u1, 2);

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_u2);
    HYPRE_ParCSRBiCGSTABSetLogging(solver_u2, 1);
    HYPRE_BiCGSTABSetMaxIter(solver_u2, 1000);
    HYPRE_BiCGSTABSetTol(solver_u2, 1e-6);
    // HYPRE_BiCGSTABSetPrintLevel(solver_u2, 2);

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_u3);
    HYPRE_ParCSRBiCGSTABSetLogging(solver_u3, 1);
    HYPRE_BiCGSTABSetMaxIter(solver_u3, 1000);
    HYPRE_BiCGSTABSetTol(solver_u3, 1e-6);
    // HYPRE_BiCGSTABSetPrintLevel(solver_u3, 2);

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_p);
    HYPRE_ParCSRBiCGSTABSetLogging(solver_p, 1);
    HYPRE_BiCGSTABSetMaxIter(solver_p, 1000);
    HYPRE_BiCGSTABSetTol(solver_p, 1e-6);
    // HYPRE_BiCGSTABSetPrintLevel(solver_p, 2);

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    HYPRE_BoomerAMGSetOldDefault(precond);
    HYPRE_BoomerAMGSetRelaxType(precond, 6);
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, 0);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    // HYPRE_BoomerAMGSetPrintLevel(precond, 2);

    HYPRE_BiCGSTABSetPrecond(
        solver_u1, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond
    );
    HYPRE_BiCGSTABSetPrecond(
        solver_u2, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond
    );
    HYPRE_BiCGSTABSetPrecond(
        solver_u3, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond
    );
    HYPRE_BiCGSTABSetPrecond(
        solver_p, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond
    );

    printf("\nHYPRE done\n");

    /****** Initialize Flow. **************************************************/

    /* Initialize from file. */
    if (init_using_file) {
        FILE *fp_init;

        fp_init = fopen_check(init_file_u1, "rb");
        fread(u1, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_init);
        fclose(fp_init);

        fp_init = fopen_check(init_file_u2, "rb");
        fread(u2, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_init);
        fclose(fp_init);

        fp_init = fopen_check(init_file_u3, "rb");
        fread(u3, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_init);
        fclose(fp_init);

        fp_init = fopen_check(init_file_p, "rb");
        fread(p, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_init);
        fclose(fp_init);

        FOR_ALL_XSTAG (i, j, k) {
            U1[i][j][k] = (u1[i][j][k]*dx[i+1] + u1[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
        }
        FOR_ALL_YSTAG (i, j, k) {
            U2[i][j][k] = (u2[i][j][k]*dy[j+1] + u2[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
        }
        FOR_ALL_ZSTAG (i, j, k) {
            U3[i][j][k] = (u3[i][j][k]*dz[k+1] + u3[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
        }
    }
    /* Initialize to uniform flow. */
    else {
        FOR_ALL_CELL (i, j, k) {
            u1[i][j][k] = 1;
        }
        FOR_ALL_XSTAG (i, j, k) {
            U1[i][j][k] = 1;
        }
    }

    calc_flux(
        Nx, Ny, Nz, dx, dy, dz,
        u1, u2, u3, U1, U2, U3,
        N1_prev, N2_prev, N3_prev
    );

    /****** Get current time. *************************************************/

    clock_gettime(CLOCK_REALTIME, &t_start);

    /****** Main loop. ********************************************************/

    for (int tstep = 1; tstep <= numtstep; tstep++) {
        /* Calculate N. */
        calc_flux(
            Nx, Ny, Nz, dx, dy, dz,
            u1, u2, u3, U1, U2, U3,
            N1, N2, N3
        );

        /* Calculate u_star. */
        FOR_ALL_CELL (i, j, k) {
            if (flag[i][j][k] != 2) {
                vector_values_u1[IDXFLAT(i, j, k)-1]
                    = -dt/2 * (3*N1[i][j][k] - N1_prev[i][j][k])
                    - dt * (p[i+1][j][k] - p[i-1][j][k]) / (xc[i+1] - xc[i-1])
                    + (1-kx_W[i]-kx_E[i]-ky_S[j]-ky_N[j]-kz_D[k]-kz_U[k]) * u1[i][j][k]
                    + kx_W[i]*u1[i-1][j][k] + kx_E[i]*u1[i+1][j][k]
                    + ky_S[j]*u1[i][j-1][k] + ky_N[j]*u1[i][j+1][k]
                    + kz_D[k]*u1[i][j][k-1] + kz_U[k]*u1[i][j][k+1];
                vector_values_u2[IDXFLAT(i, j, k)-1]
                    = -dt/2 * (3*N2[i][j][k] - N2_prev[i][j][k])
                    - dt * (p[i][j+1][k] - p[i][j-1][k]) / (yc[j+1] - yc[j-1])
                    + (1-kx_W[i]-kx_E[i]-ky_S[j]-ky_N[j]-kz_D[k]-kz_U[k]) * u2[i][j][k]
                    + kx_W[i]*u2[i-1][j][k] + kx_E[i]*u2[i+1][j][k]
                    + ky_S[j]*u2[i][j-1][k] + ky_N[j]*u2[i][j+1][k]
                    + kz_D[k]*u2[i][j][k-1] + kz_U[k]*u2[i][j][k+1];
                vector_values_u3[IDXFLAT(i, j, k)-1]
                    = -dt/2 * (3*N3[i][j][k] - N3_prev[i][j][k])
                    - dt * (p[i][j][k+1] - p[i][j][k-1]) / (zc[k+1] - zc[k-1])
                    + (1-kx_W[i]-kx_E[i]-ky_S[j]-ky_N[j]-kz_D[k]-kz_U[k]) * u3[i][j][k]
                    + kx_W[i]*u3[i-1][j][k] + kx_E[i]*u3[i+1][j][k]
                    + ky_S[j]*u3[i][j-1][k] + ky_N[j]*u3[i][j+1][k]
                    + kz_D[k]*u3[i][j][k-1] + kz_U[k]*u3[i][j][k+1];

                if (i == 1) {
                    vector_values_u1[IDXFLAT(i, j, k)-1] += 2*kx_W[i];
                }
            }
        }

        HYPRE_IJVectorSetValues(b_u1, Nx*Ny*Nz, vector_rows, vector_values_u1);
        HYPRE_IJVectorSetValues(b_u2, Nx*Ny*Nz, vector_rows, vector_values_u2);
        HYPRE_IJVectorSetValues(b_u3, Nx*Ny*Nz, vector_rows, vector_values_u3);

        HYPRE_IJVectorSetValues(x_u1, Nx*Ny*Nz, vector_rows, vector_zeros);
        HYPRE_IJVectorSetValues(x_u2, Nx*Ny*Nz, vector_rows, vector_zeros);
        HYPRE_IJVectorSetValues(x_u3, Nx*Ny*Nz, vector_rows, vector_zeros);

        HYPRE_IJVectorAssemble(b_u1);
        HYPRE_IJVectorAssemble(b_u2);
        HYPRE_IJVectorAssemble(b_u3);
        HYPRE_IJVectorAssemble(x_u1);
        HYPRE_IJVectorAssemble(x_u2);
        HYPRE_IJVectorAssemble(x_u3);

        HYPRE_IJVectorGetObject(b_u1, (void **)&par_b_u1);
        HYPRE_IJVectorGetObject(b_u2, (void **)&par_b_u2);
        HYPRE_IJVectorGetObject(b_u3, (void **)&par_b_u3);
        HYPRE_IJVectorGetObject(x_u1, (void **)&par_x_u1);
        HYPRE_IJVectorGetObject(x_u2, (void **)&par_x_u2);
        HYPRE_IJVectorGetObject(x_u3, (void **)&par_x_u3);

        HYPRE_ParCSRBiCGSTABSetup(solver_u1, parcsr_A_u1, par_b_u1, par_x_u1);
        HYPRE_ParCSRBiCGSTABSetup(solver_u2, parcsr_A_u2, par_b_u2, par_x_u2);
        HYPRE_ParCSRBiCGSTABSetup(solver_u3, parcsr_A_u3, par_b_u3, par_x_u3);

        HYPRE_ParCSRBiCGSTABSolve(solver_u1, parcsr_A_u1, par_b_u1, par_x_u1);
        HYPRE_ParCSRBiCGSTABSolve(solver_u2, parcsr_A_u2, par_b_u2, par_x_u2);
        HYPRE_ParCSRBiCGSTABSolve(solver_u3, parcsr_A_u3, par_b_u3, par_x_u3);

        HYPRE_IJVectorGetValues(x_u1, Nx*Ny*Nz, vector_rows, vector_res);
        FOR_ALL_CELL (i, j, k) {
            u1_star[i][j][k] = vector_res[IDXFLAT(i, j, k)-1];
        }
        HYPRE_IJVectorGetValues(x_u2, Nx*Ny*Nz, vector_rows, vector_res);
        FOR_ALL_CELL (i, j, k) {
            u2_star[i][j][k] = vector_res[IDXFLAT(i, j, k)-1];
        }
        HYPRE_IJVectorGetValues(x_u3, Nx*Ny*Nz, vector_rows, vector_res);
        FOR_ALL_CELL (i, j, k) {
            u3_star[i][j][k] = vector_res[IDXFLAT(i, j, k)-1];
        }

        HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver_u1, &final_res_norm);
        if (final_res_norm >= 1e-4) {
            fprintf(stderr, "u1 not converged!\n");
        }
        HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver_u2, &final_res_norm);
        if (final_res_norm >= 1e-4) {
            fprintf(stderr, "u2 not converged!\n");
        }
        HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver_u3, &final_res_norm);
        if (final_res_norm >= 1e-4) {
            fprintf(stderr, "u3 not converged!\n");
        }

        /* Calculate u_tilde. */
        FOR_ALL_CELL (i, j, k) {
            u1_tilde[i][j][k] = u1_star[i][j][k] + dt * (p[i+1][j][k] - p[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            u2_tilde[i][j][k] = u2_star[i][j][k] + dt * (p[i][j+1][k] - p[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            u3_tilde[i][j][k] = u3_star[i][j][k] + dt * (p[i][j][k+1] - p[i][j][k-1]) / (zc[k+1] - zc[k-1]);
        }
        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                u1_tilde[Nx+1][j][k] = u1_star[Nx+1][j][k] + dt * (p[Nx+1][j][k] - p[Nx][j][k]) / (xc[Nx+1] - xc[Nx]);
                u2_tilde[Nx+1][j][k] = u2_star[Nx+1][j][k] + dt * (p[Nx+1][j+1][k] - p[Nx+1][j-1][k]) / (yc[j+1] - yc[j-1]);
                u3_tilde[Nx+1][j][k] = u3_star[Nx+1][j][k] + dt * (p[Nx+1][j][k+1] - p[Nx+1][j][k-1]) / (zc[k+1] - zc[k-1]);
            }
        }

        /* Calculate U_star. */
        FOR_ALL_XSTAG (i, j, k) {
            U1_star[i][j][k] = (u1_tilde[i][j][k]*dx[i+1] + u1_tilde[i+1][j][k]*dx[i]) / (dx[i] + dx[i+1])
                - dt * (p[i+1][j][k] - p[i][j][k]) / (xc[i+1] - xc[i]);
        }
        FOR_ALL_YSTAG (i, j, k) {
            U2_star[i][j][k] = (u2_tilde[i][j][k]*dy[j+1] + u2_tilde[i][j+1][k]*dy[j]) / (dy[j] + dy[j+1])
                - dt * (p[i][j+1][k] - p[i][j][k]) / (yc[j+1] - yc[j]);
        }
        FOR_ALL_ZSTAG (i, j, k) {
            U3_star[i][j][k] = (u3_tilde[i][j][k]*dz[k+1] + u3_tilde[i][j][k+1]*dz[k]) / (dz[k] + dz[k+1])
                - dt * (p[i][j][k+1] - p[i][j][k]) / (zc[k+1] - zc[k]);
        }

        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                U1_star[0][j][k] = 1;
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                U2_star[i][0][k] = U2_star[i][Ny][k] = 0;
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U3_star[i][j][0] = U3_star[i][j][Nz] = 0;
            }
        }

        /* Calculate p_prime. */
        FOR_ALL_CELL (i, j, k) {
            if (flag[i][j][k] != 2) {
                double coeffsum = kx_W[i] + kx_E[i] + ky_S[j] + ky_N[j] + kz_D[k] + kz_U[k];
                if (i == 1) {
                    coeffsum -= kx_W[i];
                }
                if (i == Nx) {
                    coeffsum += kx_E[i];
                }
                if (j == 1) {
                    coeffsum -= ky_S[j];
                }
                if (j == Ny) {
                    coeffsum -= ky_N[j];
                }
                if (k == 1) {
                    coeffsum -= kz_D[k];
                }
                if (k == Nz) {
                    coeffsum -= kz_U[k];
                }

                vector_values_p[IDXFLAT(i, j, k)-1]
                    = -1/(2*Re*coeffsum) * (
                        (U1_star[i][j][k] - U1_star[i-1][j][k]) / dx[i]
                        + (U2_star[i][j][k] - U2_star[i][j-1][k]) / dy[j]
                        + (U3_star[i][j][k] - U3_star[i][j][k-1]) / dz[k]
                    );
            }
        }

        HYPRE_IJVectorSetValues(b_p, Nx*Ny*Nz, vector_rows, vector_values_p);
        HYPRE_IJVectorSetValues(x_p, Nx*Ny*Nz, vector_rows, vector_zeros);

        HYPRE_IJVectorAssemble(b_p);
        HYPRE_IJVectorAssemble(x_p);

        HYPRE_IJVectorGetObject(b_p, (void **)&par_b_p);
        HYPRE_IJVectorGetObject(x_p, (void **)&par_x_p);

        HYPRE_ParCSRBiCGSTABSetup(solver_p, parcsr_A_p, par_b_p, par_x_p);

        HYPRE_ParCSRBiCGSTABSolve(solver_p, parcsr_A_p, par_b_p, par_x_p);

        HYPRE_IJVectorGetValues(x_p, Nx*Ny*Nz, vector_rows, vector_res);
        FOR_ALL_CELL (i, j, k) {
            p_prime[i][j][k] = vector_res[IDXFLAT(i, j, k)-1];
        }

        HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver_p, &final_res_norm);
        if (final_res_norm >= 1e-4) {
            fprintf(stderr, "p not converged!\n");
        }

        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[0][j][k] = p_prime[1][j][k];
                p_prime[Nx+1][j][k] = -p_prime[Nx][j][k];
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_prime[i][0][k] = p_prime[i][1][k];
                p_prime[i][Ny+1][k] = p_prime[i][Ny][k];
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                p_prime[i][j][0] = p_prime[i][j][1];
                p_prime[i][j][Nz+1] = p_prime[i][j][Nz];
            }
        }

        /* Calculate p_next. */
        FOR_ALL_CELL (i, j, k) {
            p_next[i][j][k] = p[i][j][k] + p_prime[i][j][k];
        }

        /* Calculate u_next. */
        FOR_ALL_CELL (i, j, k) {
            u1_next[i][j][k] = u1_star[i][j][k] - dt * (p_prime[i+1][j][k] - p_prime[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            u2_next[i][j][k] = u2_star[i][j][k] - dt * (p_prime[i][j+1][k] - p_prime[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            u3_next[i][j][k] = u3_star[i][j][k] - dt * (p_prime[i][j][k+1] - p_prime[i][j][k-1]) / (zc[k+1] - zc[k-1]);
        }

        /* Calculate U_next. */
        FOR_ALL_XSTAG (i, j, k) {
            U1_next[i][j][k] = U1_star[i][j][k] - dt * (p_prime[i+1][j][k] - p_prime[i][j][k]) / (xc[i+1] - xc[i]);
        }
        FOR_ALL_YSTAG (i, j, k) {
            U2_next[i][j][k] = U2_star[i][j][k] - dt * (p_prime[i][j+1][k] - p_prime[i][j][k]) / (yc[j+1] - yc[j]);
        }
        FOR_ALL_ZSTAG (i, j, k) {
            U3_next[i][j][k] = U3_star[i][j][k] - dt * (p_prime[i][j][k+1] - p_prime[i][j][k]) / (zc[k+1] - zc[k]);
        }

        /* Set velocity boundary conditions. */
        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                u1_next[0][j][k] = 2 - u1_next[1][j][k];
                u2_next[0][j][k] = -u2_next[1][j][k];
                u3_next[0][j][k] = -u3_next[1][j][k];

                u1_next[Nx+1][j][k] = ((xc[Nx+1]-xc[Nx-1])*u1_next[Nx][j][k] - (xc[Nx+1]-xc[Nx])*u1_next[Nx-1][j][k]) / (xc[Nx] - xc[Nx-1]);
                u2_next[Nx+1][j][k] = ((xc[Nx+1]-xc[Nx-1])*u2_next[Nx][j][k] - (xc[Nx+1]-xc[Nx])*u2_next[Nx-1][j][k]) / (xc[Nx] - xc[Nx-1]);
                u3_next[Nx+1][j][k] = ((xc[Nx+1]-xc[Nx-1])*u3_next[Nx][j][k] - (xc[Nx+1]-xc[Nx])*u3_next[Nx-1][j][k]) / (xc[Nx] - xc[Nx-1]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                u1_next[i][0][k] = u1_next[i][1][k];
                u2_next[i][0][k] = -u2_next[i][1][k];
                u3_next[i][0][k] = -u3_next[i][1][k];

                u1_next[i][Ny+1][k] = u1_next[i][Ny][k];
                u2_next[i][Ny+1][k] = -u2_next[i][Ny][k];
                u3_next[i][Ny+1][k] = -u3_next[i][Ny][k];
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                u1_next[i][j][0] = u1_next[i][j][1];
                u2_next[i][j][0] = -u2_next[i][j][1];
                u3_next[i][j][0] = -u3_next[i][j][1];

                u1_next[i][j][Nz+1] = u1_next[i][j][Nz];
                u2_next[i][j][Nz+1] = -u2_next[i][j][Nz];
                u3_next[i][j][Nz+1] = -u3_next[i][j][Nz];
            }
        }

        /* Set pressure boundary conditions. */
        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                p_next[0][j][k] = p_next[1][j][k];
                p_next[Nx+1][j][k] = -p_next[Nx][j][k];
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int k = 1; k <= Nz; k++) {
                p_next[i][0][k] = p_next[i][1][k];
                p_next[i][Ny+1][k] = p_next[i][Ny][k];
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                p_next[i][j][0] = p_next[i][j][1];
                p_next[i][j][Nz+1] = p_next[i][j][Nz];
            }
        }

        /* Update for next time step. */
        SWAP(u1, u1_next);
        SWAP(u2, u2_next);
        SWAP(u3, u3_next);
        SWAP(U1, U1_next);
        SWAP(U2, U2_next);
        SWAP(U3, U3_next);
        SWAP(p, p_next);
        SWAP(N1_prev, N1);
        SWAP(N2_prev, N2);
        SWAP(N3_prev, N3);
    }

    /****** Calculate elapsed time. *******************************************/

    clock_gettime(CLOCK_REALTIME, &t_end);
    printf(
        "\nelapsed time: %ld ms\n",
        (t_end.tv_sec*1000+t_end.tv_nsec/1000000)
            - (t_start.tv_sec*1000+t_start.tv_nsec/1000000)
    );

    /****** Export result. ****************************************************/

    fp_out = fopen_check(output_file_u1, "wb");
    fwrite(u1, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_out);
    fclose(fp_out);

    fp_out = fopen_check(output_file_u2, "wb");
    fwrite(u2, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_out);
    fclose(fp_out);

    fp_out = fopen_check(output_file_u3, "wb");
    fwrite(u3, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_out);
    fclose(fp_out);

    fp_out = fopen_check(output_file_p, "wb");
    fwrite(p, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_out);
    fclose(fp_out);

    /****** Free memory. ******************************************************/

    HYPRE_Finalize();
    MPI_Finalize();
}

static inline HYPRE_IJMatrix create_matrix(
    const int Nx, const int Ny, const int Nz,
    const double xc[const restrict static Nx+2],
    const double yc[const restrict static Ny+2],
    const double zc[const restrict static Nz+2],
    const double lvset[const restrict static Nx+2][Ny+2][Nz+2],
    const int flag[const restrict static Nx+2][Ny+2][Nz+2],
    const double kx_W[const restrict static Nx+2],
    const double kx_E[const restrict static Nx+2],
    const double ky_S[const restrict static Ny+2],
    const double ky_N[const restrict static Ny+2],
    const double kz_D[const restrict static Nz+2],
    const double kz_U[const restrict static Nz+2],
    const int type
) {
    HYPRE_IJMatrix A;

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, 1, Nx*Ny*Nz, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    FOR_ALL_CELL (i, j, k) {
        int cur_idx = IDXFLAT(i, j, k);
        int ncols;
        int cols[9];
        double values[9];

        /* Fluid and solid cell. */
        if (flag[i][j][k] != 2) {
            cols[0] = cur_idx;
            for (int l = 0; l < 6; l++) {
                cols[l+1] = IDXFLAT(i+adj[l][0], j+adj[l][1], k+adj[l][2]);
            }
            values[0] = 1+ky_N[j]+kx_E[i]+ky_S[j]+kx_W[i]+kz_D[k]+kz_U[k];
            values[1] = -ky_N[j];
            values[2] = -kx_E[i];
            values[3] = -ky_S[j];
            values[4] = -kx_W[i];
            values[5] = -kz_D[k];
            values[6] = -kz_U[k];

            if (type == 4) {
                values[0] -= 1;
            }

            /* West (i = 1) => velocity inlet.
                * u1[0][j][k] + u1[1][j][k] = 2
                * u2[0][j][k] + u2[1][j][k] = 0
                * u3[0][j][k] + u3[1][j][k] = 0
                * p[0][j][k] = p[1][j][k] */
            if (i == 1) {
                if (type == 4) {
                    values[0] -= kx_W[i];
                }
                else {
                    values[0] += kx_W[i];
                }
                values[4] = 0;
            }

            /* East (i = Nx) => pressure outlet.
                * u1[Nx+1][j][k] is linearly extrapolated using u1[Nx-1][j][k] and
                    u1[Nx][j][k]. Same for u2 and u3.
                * p[Nx+1][j][k] + p[Nx][j][k] = 0 */
            if (i == Nx) {
                if (type == 4) {
                    values[0] -= kx_E[i];
                }
                else {
                    values[0] -= kx_E[i]*(xc[i+1]-xc[i-1])/(xc[i]-xc[i-1]);
                    values[4] += kx_E[i]*(xc[i+1]-xc[i])/(xc[i]-xc[i-1]);
                }
                values[2] = 0;
            }

            /* South (j = 1) => free-slip wall.
                * u1[i][0][k] = u1[i][1][k]
                * u2[i][0][k] + u2[i][1][k] = 0
                * u3[i][0][k] = u3[i][1][k]
                * p[i][0][k] = p[i][1][k] */
            if (j == 1) {
                if (type == 2) {
                    values[0] += ky_S[j];
                }
                else {
                    values[0] -= ky_S[j];
                }
                values[3] = 0;
            }

            /* North (j = Ny) => free-slip wall.
                * u1[i][Ny+1][k] = u1[i][Ny][k]
                * u2[i][Ny+1][k] + u2[i][Ny][k] = 0
                * u3[i][Ny+1][k] = u3[i][Ny][k]
                * p[i][Ny+1][k] = p[i][Ny][k] */
            if (j == Ny) {
                if (type == 2) {
                    values[0] += ky_N[j];
                }
                else {
                    values[0] -= ky_N[j];
                }
                values[1] = 0;
            }

            /* Upper (k = 1) => free-slip wall.
                * u1[i][j][0] = u1[i][j][1]
                * u2[i][j][0] = u2[i][j][1]
                * u3[i][j][0] + u3[i][j][1] = 0
                * p[i][j][0] = p[i][j][1] */
            if (k == 1) {
                if (type == 3) {
                    values[0] += kz_D[k];
                }
                else {
                    values[0] -= kz_D[k];
                }
                values[5] = 0;
            }

            /* Upper (k = Nz) => free-slip wall.
                * u1[i][j][Nz+1] = u1[i][j][Nz]
                * u2[i][j][Nz+1] = u2[i][j][Nz]
                * u3[i][j][Nz+1] + u3[i][j][Nz] = 0
                * p[i][j][Nz+1] = p[i][j][Nz] */
            if (k == Nz) {
                if (type == 3) {
                    values[0] += kz_U[k];
                }
                else {
                    values[0] -= kz_U[k];
                }
                values[6] = 0;
            }

            ncols = 1;
            for (int l = 1; l < 7; l++) {
                if (values[l] != 0) {
                    cols[ncols] = cols[l];
                    values[ncols] = values[l];
                    ncols++;
                }
            }

            if (type == 4) {
                for (int l = 1; l < ncols; l++) {
                    values[l] /= values[0];
                }
                values[0] = 1;
            }
        }

        /* Ghost cell. */
        else {
            int idx = -1;
            int interp_idx[8];
            double interp_coeff[8];

            get_interp_info(Nx, Ny, Nz, xc, yc, zc, lvset, i, j, k, interp_idx, interp_coeff);

            for (int l = 0; l < 8; l++) {
                if (interp_idx[l] == cur_idx) {
                    idx = l;
                    break;
                }
            }

            /* If the mirror point is not interpolated using the ghost cell
               itself. */
            if (idx == -1) {
                ncols = 9;
                cols[0] = cur_idx;
                for (int l = 0; l < 8; l++) {
                    cols[l+1] = interp_idx[l];
                }
                values[0] = 1;
                for (int l = 0; l < 8; l++) {
                    values[l+1] = interp_coeff[l];
                }
            }
            /* Otherwise. */
            else {
                ncols = 8;
                cols[0] = cur_idx;
                values[0] = 1 + interp_coeff[idx];
                int cnt = 1;
                for (int l = 0; l < 8; l++) {
                    if (l != idx) {
                        cols[cnt] = interp_idx[l];
                        values[cnt] = interp_coeff[l];
                        cnt++;
                    }
                }
            }
        }

        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cur_idx, cols, values);
    }

    HYPRE_IJMatrixAssemble(A);

    return A;
}

static void get_interp_info(
    const int Nx, const int Ny, const int Nz,
    const double xc[const restrict static Nx+2],
    const double yc[const restrict static Ny+2],
    const double zc[const restrict static Nz+2],
    const double lvset[const restrict static Nx+2][Ny+2][Nz+2],
    const int i, const int j, const int k,
    int interp_idx[const restrict static 8],
    double interp_coeff[const restrict static 8]
) {
    Vector n, m;

    n.x = (lvset[i+1][j][k] - lvset[i-1][j][k]) / (xc[i+1] - xc[i-1]);
    n.y = (lvset[i][j+1][k] - lvset[i][j-1][k]) / (yc[j+1] - yc[j-1]);
    n.z = (lvset[i][j][k+1] - lvset[i][j][k-1]) / (zc[k+1] - zc[k-1]);

    m = Vector_lincom(1, (Vector){xc[i], yc[j], zc[k]}, -2*lvset[i][j][k], n);

    const int im = lower_bound(Nx+2, xc, m.x);
    const int jm = lower_bound(Ny+2, yc, m.y);
    const int km = lower_bound(Nz+2, zc, m.z);

    /* Order of cells:
            011          111
             +----------+
        001 /|     101 /|          z
           +----------+ |          | y
           | |        | |          |/
           | +--------|-+          +------ x
           |/ 010     |/ 110
           +----------+
          000        100
    */
    interp_idx[0] = IDXFLAT(im  , jm  , km  );
    interp_idx[1] = IDXFLAT(im  , jm  , km+1);
    interp_idx[2] = IDXFLAT(im  , jm+1, km  );
    interp_idx[3] = IDXFLAT(im  , jm+1, km+1);
    interp_idx[4] = IDXFLAT(im+1, jm  , km  );
    interp_idx[5] = IDXFLAT(im+1, jm  , km+1);
    interp_idx[6] = IDXFLAT(im+1, jm+1, km  );
    interp_idx[7] = IDXFLAT(im+1, jm+1, km+1);

    const double xl = xc[im], xu = xc[im+1];
    const double yl = yc[jm], yu = yc[jm+1];
    const double zl = zc[km], zu = zc[km+1];
    const double vol = (xu - xl) * (yu - yl) * (zu - zl);

    interp_coeff[0] = (xu-m.x)*(yu-m.y)*(zu-m.z) / vol;
    interp_coeff[1] = (xu-m.x)*(yu-m.y)*(m.z-zl) / vol;
    interp_coeff[2] = (xu-m.x)*(m.y-yl)*(zu-m.z) / vol;
    interp_coeff[3] = (xu-m.x)*(m.y-yl)*(m.z-zl) / vol;
    interp_coeff[4] = (m.x-xl)*(yu-m.y)*(zu-m.z) / vol;
    interp_coeff[5] = (m.x-xl)*(yu-m.y)*(m.z-zl) / vol;
    interp_coeff[6] = (m.x-xl)*(m.y-yl)*(zu-m.z) / vol;
    interp_coeff[7] = (m.x-xl)*(m.y-yl)*(m.z-zl) / vol;
}

static FILE *fopen_check(const char *restrict filename, const char *restrict modes) {
    FILE *fp = fopen(filename, modes);
    if (!fp) {
        fprintf(stderr, "error: cannot open file \"%s\"\n", filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return fp;
}

void calc_flux(
    const int Nx, const int Ny, const int Nz,
    const double dx[const restrict static Nx+2],
    const double dy[const restrict static Ny+2],
    const double dz[const restrict static Nz+2],
    const double u1[const restrict static Nx+2][Ny+2][Nz+2],
    const double u2[const restrict static Nx+2][Ny+2][Nz+2],
    const double u3[const restrict static Nx+2][Ny+2][Nz+2],
    const double U1[const restrict static Nx+1][Ny+2][Nz+2],
    const double U2[const restrict static Nx+2][Ny+1][Nz+2],
    const double U3[const restrict static Nx+2][Ny+2][Nz+1],
    double N1[const restrict static Nx+2][Ny+2][Nz+2],
    double N2[const restrict static Nx+2][Ny+2][Nz+2],
    double N3[const restrict static Nx+2][Ny+2][Nz+2]
) {
    double u1_w, u1_e, u1_s, u1_n, u1_d, u1_u;
    double u2_w, u2_e, u2_s, u2_n, u2_d, u2_u;
    double u3_w, u3_e, u3_s, u3_n, u3_d, u3_u;

    FOR_ALL_CELL (i, j, k) {
        u1_w = (u1[i-1][j][k]*dx[i]+u1[i][j][k]*dx[i-1]) / (dx[i-1]+dx[i]);
        u2_w = (u2[i-1][j][k]*dx[i]+u2[i][j][k]*dx[i-1]) / (dx[i-1]+dx[i]);
        u3_w = (u3[i-1][j][k]*dx[i]+u3[i][j][k]*dx[i-1]) / (dx[i-1]+dx[i]);

        u1_e = (u1[i][j][k]*dx[i+1]+u1[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
        u2_e = (u2[i][j][k]*dx[i+1]+u2[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
        u3_e = (u3[i][j][k]*dx[i+1]+u3[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);

        u1_s = (u1[i][j-1][k]*dy[j]+u1[i][j][k]*dy[j-1]) / (dy[j-1]+dy[j]);
        u2_s = (u2[i][j-1][k]*dy[j]+u2[i][j][k]*dy[j-1]) / (dy[j-1]+dy[j]);
        u3_s = (u3[i][j-1][k]*dy[j]+u3[i][j][k]*dy[j-1]) / (dy[j-1]+dy[j]);

        u1_n = (u1[i][j][k]*dy[j+1]+u1[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
        u2_n = (u2[i][j][k]*dy[j+1]+u2[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
        u3_n = (u3[i][j][k]*dy[j+1]+u3[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);

        u1_d = (u1[i][j][k-1]*dz[k]+u1[i][j][k]*dz[k-1]) / (dz[k-1]+dz[k]);
        u2_d = (u2[i][j][k-1]*dz[k]+u2[i][j][k]*dz[k-1]) / (dz[k-1]+dz[k]);
        u3_d = (u3[i][j][k-1]*dz[k]+u3[i][j][k]*dz[k-1]) / (dz[k-1]+dz[k]);

        u1_u = (u1[i][j][k]*dz[k+1]+u1[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
        u2_u = (u2[i][j][k]*dz[k+1]+u2[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
        u3_u = (u3[i][j][k]*dz[k+1]+u3[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);

        /* Ni = d(U1ui)/dx + d(U2ui)/dy + d(U3ui)/dz */
        N1[i][j][k] = (U1[i][j][k]*u1_e-U1[i-1][j][k]*u1_w) / dx[i]
            + (U2[i][j][k]*u1_n-U2[i][j-1][k]*u1_s) / dy[j]
            + (U3[i][j][k]*u1_u-U3[i][j][k-1]*u1_d) / dz[k];
        N2[i][j][k] = (U1[i][j][k]*u2_e-U1[i-1][j][k]*u2_w) / dx[i]
            + (U2[i][j][k]*u2_n-U2[i][j-1][k]*u2_s) / dy[j]
            + (U3[i][j][k]*u2_u-U3[i][j][k-1]*u2_d) / dz[k];
        N3[i][j][k] = (U1[i][j][k]*u3_e-U1[i-1][j][k]*u3_w) / dx[i]
            + (U2[i][j][k]*u3_n-U2[i][j-1][k]*u3_s) / dy[j]
            + (U3[i][j][k]*u3_u-U3[i][j][k-1]*u3_d) / dz[k];
    }
}
