#include "ibm3d_setup.h"

#include "geo3d.h"
#include "utils.h"

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

static void alloc_arrays(IBMSolver *);
static void build_hypre(IBMSolver *, const double3d);
static HYPRE_IJMatrix create_matrix(IBMSolver *, const double3d, int type);
static void get_interp_info(
    IBMSolver *, const double3d,
    const int, const int, const int,
    int *restrict, double *restrict
);
static void interp_stag_vel(IBMSolver *);

IBMSolver *IBMSolver_new(const int num_process, const int rank) {
    IBMSolver *solver = calloc(1, sizeof(IBMSolver));

    solver->num_process = num_process;
    solver->rank = rank;

    solver->dx = solver->dy = solver->dz = NULL;
    solver->xc = solver->yc = solver->zc = NULL;
    solver->flag = NULL;
    solver->u1 = solver->u1_next = solver->u1_star = solver->u1_tilde = NULL;
    solver->u2 = solver->u2_next = solver->u2_star = solver->u2_tilde = NULL;
    solver->u3 = solver->u3_next = solver->u3_star = solver->u3_tilde = NULL;
    solver->U1 = solver->U1_next = solver->U1_star = NULL;
    solver->U2 = solver->U2_next = solver->U2_star = NULL;
    solver->U3 = solver->U3_next = solver->U3_star = NULL;
    solver->p = solver->p_next = solver->p_prime = NULL;
    solver->N1 = solver->N1_prev = NULL;
    solver->N2 = solver->N2_prev = NULL;
    solver->N3 = solver->N3_prev = NULL;
    solver->kx_W = solver->kx_E = NULL;
    solver->ky_S = solver->ky_N = NULL;
    solver->kz_D = solver->kz_U = NULL;
    return solver;
}

void IBMSolver_destroy(IBMSolver *solver) {
    free(solver->dx); free(solver->dy); free(solver->dz);
    free(solver->xc); free(solver->yc); free(solver->zc);
    free(solver->flag);

    free(solver->u1); free(solver->u1_next); free(solver->u1_star); free(solver->u1_tilde);
    free(solver->u2); free(solver->u2_next); free(solver->u2_star); free(solver->u2_tilde);
    free(solver->u3); free(solver->u3_next); free(solver->u3_star); free(solver->u3_tilde);

    free(solver->U1); free(solver->U1_next); free(solver->U1_star);
    free(solver->U2); free(solver->U2_next); free(solver->U2_star);
    free(solver->U3); free(solver->U3_next); free(solver->U3_star);

    free(solver->p); free(solver->p_next); free(solver->p_prime);

    free(solver->N1); free(solver->N1_prev);
    free(solver->N2); free(solver->N2_prev);
    free(solver->N3); free(solver->N3_prev);

    free(solver->kx_W); free(solver->kx_E);
    free(solver->ky_S); free(solver->ky_N);
    free(solver->kz_D); free(solver->kz_U);

    HYPRE_IJMatrixDestroy(solver->A_u1);
    HYPRE_IJMatrixDestroy(solver->A_u2);
    HYPRE_IJMatrixDestroy(solver->A_u3);
    HYPRE_IJMatrixDestroy(solver->A_p);
    HYPRE_IJVectorDestroy(solver->b);
    HYPRE_IJVectorDestroy(solver->x);

    free(solver->vector_rows); free(solver->vector_zeros);
    free(solver->vector_values);
    free(solver->vector_res);

    HYPRE_ParCSRBiCGSTABDestroy(solver->hypre_solver);
    HYPRE_BoomerAMGDestroy(solver->precond);
    HYPRE_ParCSRGMRESDestroy(solver->hypre_solver_p);
    HYPRE_BoomerAMGDestroy(solver->precond_p);

    free(solver);
}

void IBMSolver_set_grid_params(
    IBMSolver *solver,
    const int Nx_global, const int Ny, const int Nz,
    const double *restrict xf,
    const double *restrict yf,
    const double *restrict zf,
    const double Re, const double dt
) {
    solver->Nx_global = Nx_global;
    solver->Ny = Ny;
    solver->Nz = Nz;
    solver->Re = Re;
    solver->dt = dt;

    const int ilower = solver->ilower = solver->rank * Nx_global / solver->num_process + 1;
    const int iupper = solver->iupper = (solver->rank+1) * Nx_global / solver->num_process;
    const int Nx = solver->Nx = solver->iupper - solver->ilower + 1;

    /* Allocate arrays. */
    alloc_arrays(solver);

    /* Cell widths and centroid coordinates. */
    for (int i = 1; i <= Nx_global; i++) {
        solver->dx_global[i] = xf[i] - xf[i-1];
        solver->xc_global[i] = (xf[i] + xf[i-1]) / 2;
    }

    for (int i = 1; i <= Nx; i++) {
        solver->dx[i] = xf[LOCL_TO_GLOB(i)] - xf[LOCL_TO_GLOB(i)-1];
        solver->xc[i] = (xf[LOCL_TO_GLOB(i)] + xf[LOCL_TO_GLOB(i)-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        solver->dy[j] = yf[j] - yf[j-1];
        solver->yc[j] = (yf[j] + yf[j-1]) / 2;
    }
    for (int k = 1; k <= Nz; k++) {
        solver->dz[k] = zf[k] - zf[k-1];
        solver->zc[k] = (zf[k] + zf[k-1]) / 2;
    }

    /* Ghost cells */
    solver->dx_global[0] = solver->dx_global[1];
    solver->dx_global[Nx+1] = solver->dx_global[Nx];
    solver->xc_global[0] = 2*xf[0] - solver->xc_global[1];
    solver->xc_global[Nx_global+1] = 2*xf[Nx_global] - solver->xc_global[Nx_global];

    solver->dx[0] = solver->dx_global[ilower-1];
    solver->dx[Nx+1] = solver->dx_global[iupper+1];
    solver->dy[0] = solver->dy[1];
    solver->dy[Ny+1] = solver->dy[Ny];
    solver->dz[0] = solver->dz[1];
    solver->dz[Nz+1] = solver->dz[Nz];

    solver->xc[0] = solver->xc_global[ilower-1];
    solver->xc[Nx+1] = solver->xc_global[iupper+1];
    solver->yc[0] = 2*yf[0] - solver->yc[1];
    solver->yc[Ny+1] = 2*yf[Ny] - solver->yc[Ny];
    solver->zc[0] = 2*zf[0] - solver->zc[1];
    solver->zc[Nz+1] = 2*zf[Nz] - solver->zc[Nz];

    /* Calculate second order derivative coefficients */
    for (int i = 1; i <= Nx; i++) {
        solver->kx_W[i] = dt / (2*Re * (solver->xc[i] - solver->xc[i-1])*solver->dx[i]);
        solver->kx_E[i] = dt / (2*Re * (solver->xc[i+1] - solver->xc[i])*solver->dx[i]);
    }
    for (int j = 1; j <= Ny; j++) {
        solver->ky_S[j] = dt / (2*Re * (solver->yc[j] - solver->yc[j-1])*solver->dy[j]);
        solver->ky_N[j] = dt / (2*Re * (solver->yc[j+1] - solver->yc[j])*solver->dy[j]);
    }
    for (int k = 1; k <= Nz; k++) {
        solver->kz_D[k] = dt / (2*Re * (solver->zc[k] - solver->zc[k-1])*solver->dz[k]);
        solver->kz_U[k] = dt / (2*Re * (solver->zc[k+1] - solver->zc[k])*solver->dz[k]);
    }
}

void IBMSolver_set_obstacle(IBMSolver *solver, Polyhedron *poly) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    double (*const lvset)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    int (*flag)[Ny+2][Nz+2] = solver->flag;

    /* Calculate level set function. */
    Polyhedron_cpt(
        poly,
        Nx+2, Ny+2, Nz+2,
        solver->xc, solver->yc, solver->zc,
        lvset, .2
    );

    /* Calculate flag.
       * Level set function is positive           => fluid cell (flag = 1)
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
                is_ghost_cell = is_ghost_cell || (lvset[ni][nj][nk] > 0);
            }
            flag[i][j][k] = is_ghost_cell ? 2 : 0;
        }
    }

    /* Build HYPRE variables. */
    build_hypre(solver, lvset);

    free(lvset);
}

void IBMSolver_init_flow_const(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                u1[i][j][k] = 1;
                u2[i][j][k] = 0;
                u3[i][j][k] = 0;
                p[i][j][k] = 0;
            }
        }
    }

    interp_stag_vel(solver);
}

void IBMSolver_init_flow_file(
    IBMSolver *solver,
    const char *filename_u1,
    const char *filename_u2,
    const char *filename_u3,
    const char *filename_p
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    FILE *fp_u1 = fopen_check(filename_u1, "rb");
    FILE *fp_u2 = fopen_check(filename_u2, "rb");
    FILE *fp_u3 = fopen_check(filename_u3, "rb");
    FILE *fp_p = fopen_check(filename_p, "rb");

    fseek(fp_u1, sizeof(double)*(solver->ilower-1)*(Ny+2)*(Nz+2), SEEK_SET);
    fseek(fp_u2, sizeof(double)*(solver->ilower-1)*(Ny+2)*(Nz+2), SEEK_SET);
    fseek(fp_u3, sizeof(double)*(solver->ilower-1)*(Ny+2)*(Nz+2), SEEK_SET);
    fseek(fp_p, sizeof(double)*(solver->ilower-1)*(Ny+2)*(Nz+2), SEEK_SET);

    fread(solver->u1, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u1);
    fread(solver->u2, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u2);
    fread(solver->u3, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u3);
    fread(solver->p, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_p);

    fclose(fp_u1);
    fclose(fp_u2);
    fclose(fp_u3);
    fclose(fp_p);

    interp_stag_vel(solver);
}

void IBMSolver_export_results(
    IBMSolver *solver,
    const char *filename_u1,
    const char *filename_u2,
    const char *filename_u3,
    const char *filename_p
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const double (*const u1)[Ny+2][Nz+2] = solver->u1;
    const double (*const u2)[Ny+2][Nz+2] = solver->u2;
    const double (*const u3)[Ny+2][Nz+2] = solver->u3;
    const double (*const p)[Ny+2][Nz+2] = solver->p;

    if (solver->rank == 0) {
        double (*const u1_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const u2_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const u3_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const p_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));

        memcpy(u1_global, u1, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
        memcpy(u2_global, u2, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
        memcpy(u3_global, u3, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
        memcpy(p_global, p, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

        /* Receive from other processes. */
        for (int r = 1; r < solver->num_process; r++) {
            const int ilower_r = r * Nx_global / solver->num_process + 1;
            const int iupper_r = (r+1) * Nx_global / solver->num_process;
            const int Nx_r = iupper_r - ilower_r + 1;

            MPI_Recv(u1_global[ilower_r], (Nx_r+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u2_global[ilower_r], (Nx_r+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(u3_global[ilower_r], (Nx_r+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(p_global[ilower_r], (Nx_r+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        FILE *fp_u1 = fopen(filename_u1, "wb");
        FILE *fp_u2 = fopen(filename_u2, "wb");
        FILE *fp_u3 = fopen(filename_u3, "wb");
        FILE *fp_p = fopen(filename_p, "wb");

        fwrite(u1_global, sizeof(double), (Nx_global+2)*(Ny+2)*(Nz+2), fp_u1);
        fwrite(u2_global, sizeof(double), (Nx_global+2)*(Ny+2)*(Nz+2), fp_u2);
        fwrite(u3_global, sizeof(double), (Nx_global+2)*(Ny+2)*(Nz+2), fp_u3);
        fwrite(p_global, sizeof(double), (Nx_global+2)*(Ny+2)*(Nz+2), fp_p);

        fclose(fp_u1);
        fclose(fp_u2);
        fclose(fp_u3);
        fclose(fp_p);

        free(u1_global);
        free(u2_global);
        free(u3_global);
        free(p_global);
    }
    else {
        /* Send to process 0. */
        MPI_Send(u1[1], (Nx+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(u2[1], (Nx+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(u3[1], (Nx+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(p[1], (Nx+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
}

static void alloc_arrays(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    /* dx and xc contains global info. */
    solver->dx = calloc(Nx+2, sizeof(double));
    solver->dy = calloc(Ny+2, sizeof(double));
    solver->dz = calloc(Nz+2, sizeof(double));
    solver->xc = calloc(Nx+2, sizeof(double));
    solver->yc = calloc(Ny+2, sizeof(double));
    solver->zc = calloc(Nz+2, sizeof(double));

    solver->dx_global = calloc(Nx_global+2, sizeof(double));
    solver->xc_global = calloc(Nx_global+2, sizeof(double));

    /* Others contain only local info. */
    solver->flag = calloc(Nx_global+2, sizeof(int [Ny+2][Nz+2]));

    solver->u1       = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u1_next  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u1_star  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u1_tilde = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u2       = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u2_next  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u2_star  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u2_tilde = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u3       = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u3_next  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u3_star  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->u3_tilde = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    solver->U1      = calloc(Nx+1, sizeof(double [Ny+2][Nz+2]));
    solver->U1_next = calloc(Nx+1, sizeof(double [Ny+2][Nz+2]));
    solver->U1_star = calloc(Nx+1, sizeof(double [Ny+2][Nz+2]));
    solver->U2      = calloc(Nx+2, sizeof(double [Ny+1][Nz+2]));
    solver->U2_next = calloc(Nx+2, sizeof(double [Ny+1][Nz+2]));
    solver->U2_star = calloc(Nx+2, sizeof(double [Ny+1][Nz+2]));
    solver->U3      = calloc(Nx+2, sizeof(double [Ny+2][Nz+1]));
    solver->U3_next = calloc(Nx+2, sizeof(double [Ny+2][Nz+1]));
    solver->U3_star = calloc(Nx+2, sizeof(double [Ny+2][Nz+1]));

    solver->p       = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->p_next  = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->p_prime = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    solver->N1      = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->N1_prev = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->N2      = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->N2_prev = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->N3      = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    solver->N3_prev = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    solver->kx_W = calloc(Nx+2, sizeof(double));
    solver->kx_E = calloc(Nx+2, sizeof(double));
    solver->ky_S = calloc(Ny+2, sizeof(double));
    solver->ky_N = calloc(Ny+2, sizeof(double));
    solver->kz_D = calloc(Nz+2, sizeof(double));
    solver->kz_U = calloc(Nz+2, sizeof(double));
}

static void build_hypre(IBMSolver *solver, const double3d _lvset) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double (*lvset)[Ny+2][Nz+2] = _lvset;

    /* Matrices. */
    solver->A_u1 = create_matrix(solver, lvset, 1);
    solver->A_u2 = create_matrix(solver, lvset, 2);
    solver->A_u3 = create_matrix(solver, lvset, 3);
    solver->A_p = create_matrix(solver, lvset, 4);

    HYPRE_IJMatrixGetObject(solver->A_u1, (void **)&solver->parcsr_A_u1);
    HYPRE_IJMatrixGetObject(solver->A_u2, (void **)&solver->parcsr_A_u2);
    HYPRE_IJMatrixGetObject(solver->A_u3, (void **)&solver->parcsr_A_u3);
    HYPRE_IJMatrixGetObject(solver->A_p, (void **)&solver->parcsr_A_p);

    /* Vectors. */
    HYPRE_IJVectorCreate(
        MPI_COMM_WORLD,
        GLOB_CELL_IDX(1, 1, 1),
        GLOB_CELL_IDX(Nx, Ny, Nz),
        &solver->b
    );
    HYPRE_IJVectorSetObjectType(solver->b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(solver->b);

    HYPRE_IJVectorCreate(
        MPI_COMM_WORLD,
        GLOB_CELL_IDX(1, 1, 1),
        GLOB_CELL_IDX(Nx, Ny, Nz),
        &solver->x
    );
    HYPRE_IJVectorSetObjectType(solver->x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(solver->x);

    solver->vector_rows = calloc(Nx*Ny*Nz, sizeof(int));
    solver->vector_values = calloc(Nx*Ny*Nz, sizeof(double));
    solver->vector_zeros = calloc(Nx*Ny*Nz, sizeof(double));
    solver->vector_res = calloc(Nx*Ny*Nz, sizeof(double));

    for (int i = 0; i < Nx*Ny*Nz; i++) {
        solver->vector_rows[i] = GLOB_CELL_IDX(1, 1, 1) + i;
    }

    /* Solvers. */
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver->hypre_solver);
    HYPRE_BiCGSTABSetMaxIter(solver->hypre_solver, 1000);
    HYPRE_BiCGSTABSetTol(solver->hypre_solver, 1e-6);
    HYPRE_BiCGSTABSetLogging(solver->hypre_solver, 1);
    // HYPRE_BiCGSTABSetPrintLevel(solver->hypre_solver, 2);

    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver->hypre_solver_p);
    HYPRE_ParCSRGMRESSetMaxIter(solver->hypre_solver_p, 1000);
    HYPRE_ParCSRGMRESSetKDim(solver->hypre_solver_p, 10);
    HYPRE_ParCSRGMRESSetTol(solver->hypre_solver_p, 1e-6);
    HYPRE_ParCSRGMRESSetLogging(solver->hypre_solver_p, 1);
    // HYPRE_ParCSRGMRESSetPrintLevel(solver->hypre_solver_p, 2);

    HYPRE_BoomerAMGCreate(&solver->precond);
    HYPRE_BoomerAMGSetCoarsenType(solver->precond, 6);
    HYPRE_BoomerAMGSetOldDefault(solver->precond);
    HYPRE_BoomerAMGSetRelaxType(solver->precond, 6);
    HYPRE_BoomerAMGSetNumSweeps(solver->precond, 1);
    HYPRE_BoomerAMGSetTol(solver->precond, 0);
    HYPRE_BoomerAMGSetMaxIter(solver->precond, 1);
    // HYPRE_BoomerAMGSetPrintLevel(solver->precond, 1);

    HYPRE_BoomerAMGCreate(&solver->precond_p);
    HYPRE_BoomerAMGSetOldDefault(solver->precond_p);
    HYPRE_BoomerAMGSetTol(solver->precond_p, 0);
    HYPRE_BoomerAMGSetMaxIter(solver->precond_p, 1);
    HYPRE_BoomerAMGSetStrongThreshold(solver->precond_p, 0.5);
    HYPRE_BoomerAMGSetMaxRowSum(solver->precond_p, 1);
    HYPRE_BoomerAMGSetCoarsenType(solver->precond_p, 6);
    HYPRE_BoomerAMGSetNonGalerkinTol(solver->precond_p, 0.05);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver->precond_p, 0.00, 0);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver->precond_p, 0.01, 1);
    HYPRE_BoomerAMGSetAggNumLevels(solver->precond_p, 4);
    HYPRE_BoomerAMGSetNumSweeps(solver->precond_p, 1);
    HYPRE_BoomerAMGSetRelaxType(solver->precond_p, 6);
    HYPRE_BoomerAMGSetPrintLevel(solver->precond_p, 1);

    HYPRE_BiCGSTABSetPrecond(
        solver->hypre_solver,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        solver->precond
    );

    HYPRE_ParCSRGMRESSetPrecond(
        solver->hypre_solver_p,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        solver->precond_p
    );

    HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);
    HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);

    HYPRE_IJVectorAssemble(solver->b);
    HYPRE_IJVectorAssemble(solver->x);

    HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    HYPRE_ParCSRGMRESSetup(solver->hypre_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);

    MPI_Barrier(MPI_COMM_WORLD);
}

static HYPRE_IJMatrix create_matrix(IBMSolver *solver, const double3d _lvset, int type) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const double *kx_W = solver->kx_W;
    const double *kx_E = solver->kx_E;
    const double *ky_S = solver->ky_S;
    const double *ky_N = solver->ky_N;
    const double *kz_D = solver->kz_D;
    const double *kz_U = solver->kz_U;

    const double *xc = solver->xc;

    const int (*flag)[Ny+2][Nz+2] = solver->flag;
    const double (*lvset)[Ny+2][Nz+2] = _lvset;

    HYPRE_IJMatrix A;

    HYPRE_IJMatrixCreate(
        MPI_COMM_WORLD,
        GLOB_CELL_IDX(1, 1, 1), GLOB_CELL_IDX(Nx, Ny, Nz),
        GLOB_CELL_IDX(1, 1, 1), GLOB_CELL_IDX(Nx, Ny, Nz),
        &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    FOR_ALL_CELL (i, j, k) {
        int cur_idx = GLOB_CELL_IDX(i, j, k);
        int ncols;
        int cols[9];
        double values[9];

        /* Fluid and solid cell. */
        if (flag[i][j][k] != 2) {
            cols[0] = cur_idx;
            for (int l = 0; l < 6; l++) {
                cols[l+1] = GLOB_CELL_IDX(i+adj[l][0], j+adj[l][1], k+adj[l][2]);
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
            if (LOCL_TO_GLOB(i) == 1) {
                if (type == 4) {
                    values[0] -= kx_W[i];
                }
                else {
                    values[0] += kx_W[i];
                }
                values[4] = 0;
            }

            /* East (i = Nx) => pressure outlet.
                * u[Nx+1][j][k] is linearly extrapolated using u[Nx-1][j][k] and
                  u[Nx][j][k] where u = u1, u2, or u3.
                * p[Nx+1][j][k] + p[Nx][j][k] = 0 */
            if (LOCL_TO_GLOB(i) == Nx_global) {
                if (type == 4) {
                    values[0] += kx_E[i];
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

            get_interp_info(solver, lvset, i, j, k, interp_idx, interp_coeff);

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
                if (type != 4) {
                    values[0] = 1 + interp_coeff[idx];
                }
                else {
                    values[0] = 1 - interp_coeff[idx];
                }
                int cnt = 1;
                for (int l = 0; l < 8; l++) {
                    if (l != idx) {
                        cols[cnt] = interp_idx[l];
                        if (type != 4) {
                            values[cnt] = interp_coeff[l];
                        }
                        else {
                            values[cnt] = -interp_coeff[l];
                        }
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
    IBMSolver *solver, const double3d _lvset,
    const int i, const int j, const int k,
    int *restrict interp_idx, double *restrict interp_coeff
) {
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const double *xc = solver->xc;
    const double *yc = solver->yc;
    const double *zc = solver->zc;
    const double *xc_global = solver->xc_global;

    const double (*lvset)[Ny+2][Nz+2] = _lvset;

    Vector n, m;

    n.x = (lvset[i+1][j][k] - lvset[i-1][j][k]) / (xc[i+1] - xc[i-1]);
    n.y = (lvset[i][j+1][k] - lvset[i][j-1][k]) / (yc[j+1] - yc[j-1]);
    n.z = (lvset[i][j][k+1] - lvset[i][j][k-1]) / (zc[k+1] - zc[k-1]);

    m = Vector_lincom(
        1, (Vector){xc[i], yc[j], zc[k]},
        -2*lvset[i][j][k], n
    );

    const int im = upper_bound(Nx_global+2, xc_global, m.x) - 1;
    const int jm = upper_bound(Ny+2, yc, m.y) - 1;
    const int km = upper_bound(Nz+2, zc, m.z) - 1;

    /* Order of cells:
            011        111
             +----------+
        001 /|     101 /|          z
           +----------+ |          | y
           | |        | |          |/
           | +--------|-+          +------ x
           |/ 010     |/ 110
           +----------+
          000        100
    */
    interp_idx[0] = LOCL_CELL_IDX(im  , jm  , km  );
    interp_idx[1] = LOCL_CELL_IDX(im  , jm  , km+1);
    interp_idx[2] = LOCL_CELL_IDX(im  , jm+1, km  );
    interp_idx[3] = LOCL_CELL_IDX(im  , jm+1, km+1);
    interp_idx[4] = LOCL_CELL_IDX(im+1, jm  , km  );
    interp_idx[5] = LOCL_CELL_IDX(im+1, jm  , km+1);
    interp_idx[6] = LOCL_CELL_IDX(im+1, jm+1, km  );
    interp_idx[7] = LOCL_CELL_IDX(im+1, jm+1, km+1);

    const double xl = xc_global[im], xu = xc_global[im+1];
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

static void interp_stag_vel(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const dx = solver->dx;
    const double *const dy = solver->dy;
    const double *const dz = solver->dz;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;

    double (*const U1)[Ny+2][Nz+2] = solver->U1;
    double (*const U2)[Ny+1][Nz+2] = solver->U2;
    double (*const U3)[Ny+2][Nz+1] = solver->U3;

    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                U1[i][j][k] = (u1[i][j][k]*dx[i+1] + u1[i+1][j][k]*dx[i]) / (dx[i]+dx[i+1]);
            }
        }
    }
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny; j++) {
            for (int k = 0; k <= Nz+1; k++) {
        U2[i][j][k] = (u2[i][j][k]*dy[j+1] + u2[i][j+1][k]*dy[j]) / (dy[j]+dy[j+1]);
            }
        }
    }
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz; k++) {
                U3[i][j][k] = (u3[i][j][k]*dz[k+1] + u3[i][j][k+1]*dz[k]) / (dz[k]+dz[k+1]);
            }
        }
    }
}
