#include "ibm3d_setup.h"

#include "geo3d.h"
#include "utils.h"
#include "math.h"

#include <string.h>

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
static void calc_lvset_flag(IBMSolver *);
static void build_hypre(IBMSolver *);
static HYPRE_IJMatrix create_matrix(IBMSolver *, int);

static void get_interp_info(
    IBMSolver *,
    const int, const int, const int,
    int *restrict, double *restrict
);
static bool isperiodic(IBMSolverBCType);

IBMSolver *IBMSolver_new(const int num_process, const int rank) {
    IBMSolver *solver = calloc(1, sizeof(IBMSolver));

    solver->num_process = num_process;
    solver->rank = rank;

    solver->dx = solver->dy = solver->dz = NULL;
    solver->xc = solver->yc = solver->zc = NULL;
    solver->flag = NULL;
    solver->lvset = NULL;
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

    for (int i = 0; i < 6; i++) {
        solver->bc_val[i] = NAN;
    }

    return solver;
}

void IBMSolver_destroy(IBMSolver *solver) {
    free(solver->dx); free(solver->dy); free(solver->dz);
    free(solver->xc); free(solver->yc); free(solver->zc);
    free(solver->dx_global); free(solver->xc_global);

    free(solver->flag); free(solver->lvset);

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

    HYPRE_ParCSRBiCGSTABDestroy(solver->linear_solver);
    HYPRE_BoomerAMGDestroy(solver->precond);

    switch (solver->linear_solver_type) {
    case SOLVER_AMG:
        HYPRE_BoomerAMGDestroy(solver->linear_solver_p);
        break;
    case SOLVER_PCG:
        HYPRE_ParCSRPCGDestroy(solver->linear_solver_p);
        break;
    case SOLVER_BiCGSTAB:
        HYPRE_ParCSRBiCGSTABDestroy(solver->linear_solver_p);
        break;
    case SOLVER_GMRES:
        HYPRE_ParCSRGMRESDestroy(solver->linear_solver_p);
        break;
    default:;
    }
    switch (solver->precond_type) {
    case PRECOND_NONE:
        break;
    case PRECOND_AMG:
        HYPRE_BoomerAMGDestroy(solver->precond_p);
        break;
    default:;
    }

    free(solver);
}

void IBMSolver_set_grid(
    IBMSolver *solver,
    const int Nx_global, const int Ny, const int Nz,
    const double *restrict xf,
    const double *restrict yf,
    const double *restrict zf
) {
    solver->Nx_global = Nx_global;
    solver->Ny = Ny;
    solver->Nz = Nz;

    solver->ilower = solver->rank * Nx_global / solver->num_process + 1;
    solver->iupper = (solver->rank+1) * Nx_global / solver->num_process;
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
}

void IBMSolver_set_params(IBMSolver *solver, const double Re, const double dt) {
    solver->Re = Re;
    solver->dt = dt;
}

void IBMSolver_set_bc(
    IBMSolver *solver,
    IBMSolverDirection direction,
    IBMSolverBCType bc_type,
    double bc_val
) {
    for (int i = 0; i < 6; i++) {
        if (direction & (1 << i)) {
            solver->bc_type[i] = bc_type;
            solver->bc_val[i] = bc_val;
        }
    }
}

void IBMSolver_set_obstacle(IBMSolver *solver, Polyhedron *poly) {
    solver->poly = poly;
}

void IBMSolver_set_linear_solver(
    IBMSolver *solver,
    IBMSolverLinearSolverType linear_solver_type,
    IBMSolverPrecondType precond_type,
    const double tol
) {
    solver->linear_solver_type = linear_solver_type;
    solver->precond_type = precond_type;
    solver->tol = tol;
}

void IBMSolver_assemble(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const int ilower = solver->ilower;
    const int iupper = solver->iupper;

    const double Re = solver->Re;
    const double dt = solver->dt;

    double *const dx_global = solver->dx_global;
    double *const xc_global = solver->xc_global;

    double *const dx = solver->dx;
    double *const dy = solver->dy;
    double *const dz = solver->dz;

    double *const xc = solver->xc;
    double *const yc = solver->yc;
    double *const zc = solver->zc;

    if (
        solver->rank == 0
        && (
            isperiodic(solver->bc_type[0]) != isperiodic(solver->bc_type[2])
            || isperiodic(solver->bc_type[1]) != isperiodic(solver->bc_type[3])
            || isperiodic(solver->bc_type[4]) != isperiodic(solver->bc_type[5])
        )
    ) {
        printf("Inconsistent periodic boundary condition\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Ghost cells */
    dx_global[0]
        = isperiodic(solver->bc_type[3]) ? dx_global[Nx_global] : dx_global[1];
    dx_global[Nx+1]
        = isperiodic(solver->bc_type[1]) ? dx_global[1] : dx_global[Nx];
    xc_global[0] = xc_global[1] - (dx_global[0] + dx_global[1]) / 2;
    xc_global[Nx_global+1] = xc_global[Nx_global] + (dx_global[Nx_global] + dx_global[Nx_global+1]) / 2;

    dx[0] = dx_global[ilower-1];
    dx[Nx+1] = dx_global[iupper+1];
    dy[0] = isperiodic(solver->bc_type[2]) ? dy[Ny] : dy[1];
    dy[Ny+1] = isperiodic(solver->bc_type[0]) ? dy[1] : dy[Ny];
    dz[0] = isperiodic(solver->bc_type[4]) ? dz[Nz] : dz[1];
    dz[Nz+1] = isperiodic(solver->bc_type[5]) ? dz[1] : dz[Nz];

    xc[0] = xc_global[ilower-1];
    xc[Nx+1] = xc_global[iupper+1];
    yc[0] = yc[1] - (dy[0] + dy[1]) / 2;
    yc[Ny+1] = yc[Ny] + (dy[Ny] + dy[Ny+1]) / 2;
    zc[0] = zc[1] - (dz[0] + dz[1]) / 2;
    zc[Nz+1] = zc[Nz] + (dz[Nz] + dz[Nz+1]) / 2;

    /* Calculate second order derivative coefficients */
    for (int i = 1; i <= Nx; i++) {
        solver->kx_W[i] = dt / (2*Re * (xc[i] - xc[i-1])*dx[i]);
        solver->kx_E[i] = dt / (2*Re * (xc[i+1] - xc[i])*dx[i]);
    }
    for (int j = 1; j <= Ny; j++) {
        solver->ky_S[j] = dt / (2*Re * (yc[j] - yc[j-1])*dy[j]);
        solver->ky_N[j] = dt / (2*Re * (yc[j+1] - yc[j])*dy[j]);
    }
    for (int k = 1; k <= Nz; k++) {
        solver->kz_D[k] = dt / (2*Re * (zc[k] - zc[k-1])*dz[k]);
        solver->kz_U[k] = dt / (2*Re * (zc[k+1] - zc[k])*dz[k]);
    }

    /* Calculate level set function and flag. */
    calc_lvset_flag(solver);

    /* Build HYPRE variables. */
    build_hypre(solver);
}

void IBMSolver_init_flow_const(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    solver->iter = 0;

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

    solver->iter = 0;

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
}

void IBMSolver_init_flow_func(
    IBMSolver *solver,
    IBMSolverInitFunc initfunc_u1,
    IBMSolverInitFunc initfunc_u2,
    IBMSolverInitFunc initfunc_u3,
    IBMSolverInitFunc initfunc_p
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *const xc = solver->xc;
    const double *const yc = solver->yc;
    const double *const zc = solver->zc;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    FOR_ALL_CELL(i, j, k) {
        u1[i][j][k] = initfunc_u1(xc[i], yc[j], zc[k]);
        u2[i][j][k] = initfunc_u2(xc[i], yc[j], zc[k]);
        u3[i][j][k] = initfunc_u3(xc[i], yc[j], zc[k]);
        p[i][j][k] = initfunc_p(xc[i], yc[j], zc[k]);
    }
}

void IBMSolver_set_autosave(
    IBMSolver *solver,
    const char *filename_u1,
    const char *filename_u2,
    const char *filename_u3,
    const char *filename_p,
    int period
) {
    solver->autosave_u1 = filename_u1;
    solver->autosave_u2 = filename_u2;
    solver->autosave_u3 = filename_u3;
    solver->autosave_p = filename_p;
    solver->autosave_period = period;
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

    char ext_u1[100], ext_u2[100], ext_u3[100], ext_p[100];

    /* Concatenate extension. */
    snprintf(ext_u1, 100, "%s.out", filename_u1);
    snprintf(ext_u2, 100, "%s.out", filename_u2);
    snprintf(ext_u3, 100, "%s.out", filename_u3);
    snprintf(ext_p, 100, "%s.out", filename_p);

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

        FILE *fp_u1 = fopen_check(ext_u1, "wb");
        FILE *fp_u2 = fopen_check(ext_u2, "wb");
        FILE *fp_u3 = fopen_check(ext_u3, "wb");
        FILE *fp_p = fopen_check(ext_p, "wb");

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

void IBMSolver_export_lvset(IBMSolver *solver, const char *filename) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const double (*const lvset)[Ny+2][Nz+2] = solver->lvset;

    if (solver->rank == 0) {
        double (*const lvset_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));

        memcpy(lvset_global, lvset, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

        /* Receive from other processes. */
        for (int r = 1; r < solver->num_process; r++) {
            const int ilower_r = r * Nx_global / solver->num_process + 1;
            const int iupper_r = (r+1) * Nx_global / solver->num_process;
            const int Nx_r = iupper_r - ilower_r + 1;

            MPI_Recv(lvset_global[ilower_r], (Nx_r+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        FILE *fp = fopen_check(filename, "wb");

        fwrite(lvset_global, sizeof(double), (Nx_global+2)*(Ny+2)*(Nz+2), fp);

        fclose(fp);

        free(lvset_global);
    }
    else {
        /* Send to process 0. */
        MPI_Send(lvset[1], (Nx+1)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void IBMSolver_export_flag(IBMSolver *solver, const char *filename) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const int (*const flag)[Ny+2][Nz+2] = solver->flag;

    if (solver->rank == 0) {
        int (*const flag_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(int [Ny+2][Nz+2]));

        memcpy(flag_global, flag, sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));

        /* Receive from other processes. */
        for (int r = 1; r < solver->num_process; r++) {
            const int ilower_r = r * Nx_global / solver->num_process + 1;
            const int iupper_r = (r+1) * Nx_global / solver->num_process;
            const int Nx_r = iupper_r - ilower_r + 1;

            MPI_Recv(flag_global[ilower_r], (Nx_r+1)*(Ny+2)*(Nz+2), MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        FILE *fp = fopen_check(filename, "wb");

        fwrite(flag_global, sizeof(int), (Nx_global+2)*(Ny+2)*(Nz+2), fp);

        fclose(fp);

        free(flag_global);
    }
    else {
        /* Send to process 0. */
        MPI_Send(flag[1], (Nx+1)*(Ny+2)*(Nz+2), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

static void alloc_arrays(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    /* dx_global and xc_global contain global info. Others contain only local
       info. */

    solver->dx = calloc(Nx+2, sizeof(double));
    solver->dy = calloc(Ny+2, sizeof(double));
    solver->dz = calloc(Nz+2, sizeof(double));
    solver->xc = calloc(Nx+2, sizeof(double));
    solver->yc = calloc(Ny+2, sizeof(double));
    solver->zc = calloc(Nz+2, sizeof(double));

    solver->dx_global = calloc(Nx_global+2, sizeof(double));
    solver->xc_global = calloc(Nx_global+2, sizeof(double));

    solver->flag = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));
    solver->lvset = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

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

    solver->p_coeffsum = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
}

static void calc_lvset_flag(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    int (*flag)[Ny+2][Nz+2] = solver->flag;
    double (*const lvset)[Ny+2][Nz+2] = solver->lvset;

    /* No obstacle: every cell is fluid cell. */
    if (solver->poly == NULL) {
        FOR_ALL_CELL (i, j, k) {
            flag[i][j][k] = 1;
        }
        return;
    }

    /* Calculate level set function. */
    Polyhedron_cpt(
        solver->poly,
        Nx+2, Ny+2, Nz+2,
        solver->xc, solver->yc, solver->zc,
        lvset, .5
    );

    /* Calculate flag.
       * Level set function is positive or zero.  => fluid cell
       * Level set function if negative and at
         least one adjacent cell is fluid cell.   => ghost cell
       * Otherwise.                               => solid cell */
    FOR_ALL_CELL (i, j, k) {
        if (lvset[i][j][k] >= 0) {
            flag[i][j][k] = FLAG_FLUID;
        }
        else {
            bool is_ghost_cell = false;
            for (int l = 0; l < 6; l++) {
                int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                is_ghost_cell = is_ghost_cell || (lvset[ni][nj][nk] >= 0);
            }
            flag[i][j][k] = is_ghost_cell ? FLAG_GHOST : FLAG_SOLID;
        }
    }

    /* Exchange flag between the adjacent processes. */
    if (solver->rank != solver->num_process-1) {
        /* Send to next process. */
        MPI_Send(flag[Nx], (Ny+2)*(Nz+2), MPI_INT, solver->rank+1, 0, MPI_COMM_WORLD);
    }
    if (solver->rank != 0) {
        /* Receive from previous process. */
        MPI_Recv(flag[0], (Ny+2)*(Nz+2), MPI_INT, solver->rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send to previous process. */
        MPI_Send(flag[1], (Ny+2)*(Nz+2), MPI_INT, solver->rank-1, 0, MPI_COMM_WORLD);
    }
    if (solver->rank != solver->num_process-1) {
        /* Receive from next process. */
        MPI_Recv(flag[Nx+1], (Ny+2)*(Nz+2), MPI_INT, solver->rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static void build_hypre(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    /* Matrices. */
    solver->A_u1 = create_matrix(solver, 1);
    solver->A_u2 = create_matrix(solver, 2);
    solver->A_u3 = create_matrix(solver, 3);
    solver->A_p = create_matrix(solver, 4);

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

    /* Set velocity solver. */
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver->linear_solver);
    HYPRE_BiCGSTABSetMaxIter(solver->linear_solver, 1000);
    HYPRE_BiCGSTABSetTol(solver->linear_solver, 1e-6);
    HYPRE_BiCGSTABSetLogging(solver->linear_solver, 1);
    // HYPRE_BiCGSTABSetPrintLevel(solver->hypre_solver, 2);

    HYPRE_BoomerAMGCreate(&solver->precond);
    HYPRE_BoomerAMGSetCoarsenType(solver->precond, 6);
    HYPRE_BoomerAMGSetOldDefault(solver->precond);
    HYPRE_BoomerAMGSetRelaxType(solver->precond, 6);
    HYPRE_BoomerAMGSetNumSweeps(solver->precond, 1);
    HYPRE_BoomerAMGSetTol(solver->precond, 0);
    HYPRE_BoomerAMGSetMaxIter(solver->precond, 1);
    // HYPRE_BoomerAMGSetPrintLevel(solver->precond, 1);

    HYPRE_BiCGSTABSetPrecond(
        solver->linear_solver,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        solver->precond
    );

    /* Set pressure solver. */
    switch (solver->linear_solver_type) {
    case SOLVER_AMG:
        if (solver->precond_type != PRECOND_NONE && solver->rank == 0) {
            printf("\nCannot use preconditioner with BoomerAMG solver\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        HYPRE_BoomerAMGCreate(&solver->linear_solver_p);
        HYPRE_BoomerAMGSetOldDefault(solver->linear_solver_p);
        HYPRE_BoomerAMGSetTol(solver->linear_solver_p, solver->tol);
        HYPRE_BoomerAMGSetMaxIter(solver->linear_solver_p, 1000);
        HYPRE_BoomerAMGSetMaxRowSum(solver->linear_solver_p, 1);
        HYPRE_BoomerAMGSetCoarsenType(solver->linear_solver_p, 6);
        HYPRE_BoomerAMGSetNonGalerkinTol(solver->linear_solver_p, 0.05);
        HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver->linear_solver_p, 0.00, 0);
        HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver->linear_solver_p, 0.01, 1);
        HYPRE_BoomerAMGSetAggNumLevels(solver->linear_solver_p, 1);
        HYPRE_BoomerAMGSetNumSweeps(solver->linear_solver_p, 1);
        HYPRE_BoomerAMGSetRelaxType(solver->linear_solver_p, 6);
        HYPRE_BoomerAMGSetLogging(solver->linear_solver_p, 1);
        // HYPRE_BoomerAMGSetPrintLevel(solver->linear_solver_p, 3);
        break;
    case SOLVER_PCG:
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver->linear_solver_p);
        HYPRE_ParCSRPCGSetTol(solver->linear_solver_p, solver->tol);
        HYPRE_ParCSRPCGSetMaxIter(solver->linear_solver_p, 1000);
        HYPRE_ParCSRPCGSetLogging(solver->linear_solver_p, 1);
        // HYPRE_ParCSRPCGSetPrintLevel(solver->linear_solver_p, 2);
        break;
    case SOLVER_BiCGSTAB:
        HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver->linear_solver_p);
        HYPRE_ParCSRBiCGSTABSetTol(solver->linear_solver_p, solver->tol);
        HYPRE_ParCSRBiCGSTABSetMaxIter(solver->linear_solver_p, 1000);
        HYPRE_ParCSRBiCGSTABSetLogging(solver->linear_solver_p, 1);
        // HYPRE_ParCSRBiCGSTABSetPrintLevel(solver->linear_solver_p, 2);
        break;
    case SOLVER_GMRES:
        HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver->linear_solver_p);
        HYPRE_ParCSRGMRESSetMaxIter(solver->linear_solver_p, 1000);
        HYPRE_ParCSRGMRESSetKDim(solver->linear_solver_p, 10);
        HYPRE_ParCSRGMRESSetTol(solver->linear_solver_p, solver->tol);
        HYPRE_ParCSRGMRESSetLogging(solver->linear_solver_p, 1);
        // HYPRE_ParCSRGMRESSetPrintLevel(solver->hypre_solver_p, 2);
        break;
    default:
        if (solver->rank == 0) {
            printf("\nUnknown linear solver type\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    switch (solver->precond_type) {
    case PRECOND_NONE:
        solver->precond_p = NULL;
        break;
    case PRECOND_AMG:
        HYPRE_BoomerAMGCreate(&solver->precond_p);
        HYPRE_BoomerAMGSetOldDefault(solver->precond_p);
        HYPRE_BoomerAMGSetTol(solver->precond_p, 0);
        HYPRE_BoomerAMGSetMaxIter(solver->precond_p, 1);
        HYPRE_BoomerAMGSetMaxRowSum(solver->precond_p, 1);
        HYPRE_BoomerAMGSetCoarsenType(solver->precond_p, 6);
        HYPRE_BoomerAMGSetNonGalerkinTol(solver->precond_p, 0.05);
        HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver->precond_p, 0.00, 0);
        HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver->precond_p, 0.01, 1);
        HYPRE_BoomerAMGSetAggNumLevels(solver->precond_p, 1);
        HYPRE_BoomerAMGSetNumSweeps(solver->precond_p, 1);
        HYPRE_BoomerAMGSetRelaxType(solver->precond_p, 6);
        HYPRE_BoomerAMGSetPrintLevel(solver->precond_p, 1);

        switch (solver->linear_solver_type) {
        case SOLVER_PCG:
            HYPRE_ParCSRPCGSetPrecond(
                solver->linear_solver_p,
                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                solver->precond_p
            );
            break;
        case SOLVER_BiCGSTAB:
            HYPRE_ParCSRBiCGSTABSetPrecond(
                solver->linear_solver_p,
                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                solver->precond_p
            );
            break;
        case SOLVER_GMRES:
            HYPRE_ParCSRGMRESSetPrecond(
                solver->linear_solver_p,
                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                solver->precond_p
            );
            break;
        default:;
        }
        break;
    default:
        if (solver->rank == 0) {
            printf("\nUnknown preconditioner type\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    HYPRE_IJVectorSetValues(solver->b, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);
    HYPRE_IJVectorSetValues(solver->x, Nx*Ny*Nz, solver->vector_rows, solver->vector_zeros);

    HYPRE_IJVectorAssemble(solver->b);
    HYPRE_IJVectorAssemble(solver->x);

    HYPRE_IJVectorGetObject(solver->b, (void **)&solver->par_b);
    HYPRE_IJVectorGetObject(solver->x, (void **)&solver->par_x);

    switch (solver->linear_solver_type) {
    case SOLVER_AMG:
        HYPRE_BoomerAMGSetup(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    case SOLVER_PCG:
        HYPRE_ParCSRPCGSetup(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    case SOLVER_BiCGSTAB:
        HYPRE_ParCSRBiCGSTABSetup(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    case SOLVER_GMRES:
        HYPRE_ParCSRGMRESSetup(solver->linear_solver_p, solver->parcsr_A_p, solver->par_b, solver->par_x);
        break;
    default:;
    }
}

static HYPRE_IJMatrix create_matrix(IBMSolver *solver, int type) {
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
    const double *yc = solver->yc;
    const double *zc = solver->zc;

    const int (*flag)[Ny+2][Nz+2] = solver->flag;

    double (*p_coeffsum)[Ny+2][Nz+2] = solver->p_coeffsum;

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

        /* Fluid cell. */
        if (flag[i][j][k] == FLAG_FLUID) {
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

            /* Boundary cells: a coefficient is added to values[0] for dirichlet
               boundary condition and is subtracted from for neumann boundary
               condition. Extrapolation is somewhat more complex. */

            /* West (i = 1) */
            if (LOCL_TO_GLOB(i) == 1) {
                switch (solver->bc_type[3]) {
                case BC_VELOCITY_INLET:
                case BC_STATIONARY_WALL:
                    values[4] = 0;
                    if (type != 4) values[0] += kx_W[i];
                    else           values[0] -= kx_W[i];
                    break;
                case BC_PRESSURE_OUTLET:
                    values[4] = 0;
                    if (type == 4) values[0] += kx_W[i];
                    else {
                        values[0] -= kx_W[i]*(xc[i+1]-xc[i-1])/(xc[i+1]-xc[i]);
                        values[2] += kx_W[i]*(xc[i]-xc[i-1])/(xc[i+1]-xc[i]);
                    }
                    break;
                case BC_FREE_SLIP_WALL:
                    values[4] = 0;
                    if (type == 1) values[0] += kx_W[i];
                    else           values[0] -= kx_W[i];
                    break;
                case BC_ALL_PERIODIC:
                    cols[4] = LOCL_CELL_IDX(Nx_global, j, k);
                    break;
                case BC_VELOCITY_PERIODIC:
                    if (type != 4) cols[4] = LOCL_CELL_IDX(Nx_global, j, k);
                    else {
                        values[4] = 0;
                        values[0] += kx_W[i];
                    }
                    break;
                }
            }

            /* East (i = Nx_global) */
            if (LOCL_TO_GLOB(i) == Nx_global) {
                switch (solver->bc_type[1]) {
                case BC_VELOCITY_INLET:
                case BC_STATIONARY_WALL:
                    values[2] = 0;
                    if (type != 4) values[0] += kx_E[i];
                    else           values[0] -= kx_E[i];
                    break;
                case BC_PRESSURE_OUTLET:
                    values[2] = 0;
                    if (type == 4) values[0] += kx_E[i];
                    else {
                        values[0] -= kx_E[i]*(xc[i+1]-xc[i-1])/(xc[i]-xc[i-1]);
                        values[4] += kx_E[i]*(xc[i+1]-xc[i])/(xc[i]-xc[i-1]);
                    }
                    break;
                case BC_FREE_SLIP_WALL:
                    values[2] = 0;
                    if (type == 1) values[0] += kx_E[i];
                    else           values[0] -= kx_E[i];
                    break;
                case BC_ALL_PERIODIC:
                    cols[2] = LOCL_CELL_IDX(1, j, k);
                    break;
                case BC_VELOCITY_PERIODIC:
                    if (type != 4) cols[2] = LOCL_CELL_IDX(1, j, k);
                    else {
                        values[2] = 0;
                        values[0] += kx_E[i];
                    }
                    break;
                }
            }

            /* South (j = 1) */
            if (j == 1) {
                switch (solver->bc_type[2]) {
                case BC_VELOCITY_INLET:
                case BC_STATIONARY_WALL:
                    values[3] = 0;
                    if (type != 4) values[0] += ky_S[j];
                    else           values[0] -= ky_S[j];
                    break;
                case BC_PRESSURE_OUTLET:
                    values[3] = 0;
                    if (type == 4) values[0] += ky_S[j];
                    else {
                        values[0] -= ky_S[j]*(yc[j+1]-yc[j-1])/(yc[j+1]-yc[j]);
                        values[1] += ky_S[j]*(yc[j]-yc[j-1])/(yc[j+1]-yc[j]);
                    }
                    break;
                case BC_FREE_SLIP_WALL:
                    values[3] = 0;
                    if (type == 2) values[0] += ky_S[j];
                    else           values[0] -= ky_S[j];
                    break;
                case BC_ALL_PERIODIC:
                    cols[3] = GLOB_CELL_IDX(i, Ny, k);
                    break;
                case BC_VELOCITY_PERIODIC:
                    if (type != 4) cols[3] = GLOB_CELL_IDX(i, Ny, k);
                    else {
                        values[3] = 0;
                        values[0] += ky_S[j];
                    }
                    break;
                }
            }

            /* North (j = Ny) */
            if (j == Ny) {
                switch (solver->bc_type[0]) {
                case BC_VELOCITY_INLET:
                case BC_STATIONARY_WALL:
                    values[1] = 0;
                    if (type != 4) values[0] += ky_N[j];
                    else           values[0] -= ky_N[j];
                    break;
                case BC_PRESSURE_OUTLET:
                    values[1] = 0;
                    if (type == 4) values[0] += ky_N[j];
                    else {
                        values[0] -= ky_N[j]*(yc[j+1]-yc[j-1])/(yc[j]-yc[j-1]);
                        values[3] += ky_N[j]*(yc[j+1]-yc[j])/(yc[j]-yc[j-1]);
                    }
                    break;
                case BC_FREE_SLIP_WALL:
                    values[1] = 0;
                    if (type == 2) values[0] += ky_N[j];
                    else           values[0] -= ky_N[j];
                    break;
                case BC_ALL_PERIODIC:
                    cols[1] = GLOB_CELL_IDX(i, 1, k);
                    break;
                case BC_VELOCITY_PERIODIC:
                    if (type != 4) cols[1] = GLOB_CELL_IDX(i, 1, k);
                    else {
                        values[1] = 0;
                        values[0] += ky_N[j];
                    }
                    break;
                }
            }

            /* Down (k = 1) */
            if (k == 1) {
                switch (solver->bc_type[4]) {
                case BC_VELOCITY_INLET:
                case BC_STATIONARY_WALL:
                    values[5] = 0;
                    if (type != 4) values[0] += kz_D[k];
                    else           values[0] -= kz_D[k];
                    break;
                case BC_PRESSURE_OUTLET:
                    values[5] = 0;
                    if (type == 4) values[0] += kz_D[k];
                    else {
                        values[0] -= kz_D[k]*(zc[k+1]-zc[k-1])/(zc[k+1]-zc[k]);
                        values[6] += kz_D[k]*(zc[k]-zc[k-1])/(zc[k+1]-zc[k]);
                    }
                    break;
                case BC_FREE_SLIP_WALL:
                    values[5] = 0;
                    if (type == 3) values[0] += kz_D[k];
                    else           values[0] -= kz_D[k];
                    break;
                case BC_ALL_PERIODIC:
                    cols[5] = GLOB_CELL_IDX(i, j, Nz);
                    break;
                case BC_VELOCITY_PERIODIC:
                    if (type != 4) cols[5] = GLOB_CELL_IDX(i, j, Nz);
                    else {
                        values[5] = 0;
                        values[0] += kz_D[k];
                    }
                    break;
                }
            }

            /* Up (k = Nz) */
            if (k == Nz) {
                switch (solver->bc_type[5]) {
                case BC_VELOCITY_INLET:
                case BC_STATIONARY_WALL:
                    values[6] = 0;
                    if (type != 4) values[0] += kz_U[k];
                    else           values[0] -= kz_U[k];
                    break;
                case BC_PRESSURE_OUTLET:
                    values[6] = 0;
                    if (type == 4) values[0] += kz_U[k];
                    else {
                        values[0] -= kz_U[k]*(zc[k+1]-zc[k-1])/(zc[k]-zc[k-1]);
                        values[5] += kz_U[k]*(zc[k+1]-zc[k])/(zc[k]-zc[k-1]);
                    }
                    break;
                case BC_FREE_SLIP_WALL:
                    values[6] = 0;
                    if (type == 3) values[0] += kz_U[k];
                    else           values[0] -= kz_U[k];
                    break;
                case BC_ALL_PERIODIC:
                    cols[6] = GLOB_CELL_IDX(i, j, 1);
                    break;
                case BC_VELOCITY_PERIODIC:
                    if (type != 4) cols[6] = GLOB_CELL_IDX(i, j, 1);
                    else {
                        values[6] = 0;
                        values[0] += kz_U[k];
                    }
                    break;
                }
            }

            /* Remove zero elements. */
            ncols = 1;
            for (int l = 1; l < 7; l++) {
                if (values[l] != 0) {
                    cols[ncols] = cols[l];
                    values[ncols] = values[l];
                    ncols++;
                }
            }

            /* Normalize pressure equation. */
            if (type == 4) {
                p_coeffsum[i][j][k] = values[0];
                for (int l = 1; l < ncols; l++) {
                    values[l] /= values[0];
                }
                values[0] = 1;
            }
        }

        /* Ghost cell. */
        else if (flag[i][j][k] == FLAG_GHOST) {
            int idx = -1;
            int interp_idx[8];
            double interp_coeff[8];

            get_interp_info(solver, i, j, k, interp_idx, interp_coeff);

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
                    if (type != 4) {
                        values[l+1] = interp_coeff[l];
                    }
                    else {
                        values[l+1] = -interp_coeff[l];
                    }
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

        /* Solid cell. */
        else if (flag[i][j][k] == FLAG_SOLID) {
            ncols = 1;
            cols[0] = cur_idx;
            values[0] = 1;
        }

        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cur_idx, cols, values);
    }

    HYPRE_IJMatrixAssemble(A);

    return A;
}

static void get_interp_info(
    IBMSolver *solver,
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

    const double (*lvset)[Ny+2][Nz+2] = solver->lvset;

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
    for (int l = 0; l < 8; l++) {
        interp_idx[l] = LOCL_CELL_IDX(im + !!(l & 4), jm + !!(l & 2), km + !!(l & 1));
    }

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

static bool isperiodic(IBMSolverBCType type) {
    switch (type) {
    case BC_ALL_PERIODIC:
    case BC_VELOCITY_PERIODIC:
        return true;
    default:
        return false;
    }
}
