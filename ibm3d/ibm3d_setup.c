#include "ibm3d.h"

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

IBMSolver *IBMSolver_new(void) {
    IBMSolver *solver = calloc(1, sizeof(IBMSolver));
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

    free(solver);
}

void IBMSolver_set_grid_params(
    IBMSolver *solver,
    const int Nx, const int Ny, const int Nz,
    const double *restrict xf,
    const double *restrict yf,
    const double *restrict zf,
    const double Re, const double dt
) {
    solver->Nx = Nx;
    solver->Ny = Ny;
    solver->Nz = Nz;
    solver->Re = Re;
    solver->dt = dt;

    /* Allocate arrays. */
    alloc_arrays(solver);

    /* Cell widths and centroid coordinates. */
    for (int i = 1; i <= Nx; i++) {
        solver->dx[i] = xf[i] - xf[i-1];
        solver->xc[i] = (xf[i] + xf[i-1]) / 2;
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
    solver->dx[0] = solver->dx[1];
    solver->dx[Nx+1] = solver->dx[Nx];
    solver->dy[0] = solver->dy[1];
    solver->dy[Ny+1] = solver->dy[Ny];
    solver->dz[0] = solver->dz[1];
    solver->dz[Nz+1] = solver->dz[Nz];

    solver->xc[0] = 2*xf[0] - solver->xc[1];
    solver->xc[Nx+1] = 2*xf[Nx] - solver->xc[Nx];
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

    printf("set grid done\n");
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
        lvset, .5
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

    printf("set obstacle done\n");

    /* Build HYPRE variables. */
    build_hypre(solver, lvset);

    free(lvset);

    printf("hypre done\n");
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
    FILE *fp_u1, FILE *fp_u2, FILE *fp_u3, FILE *fp_p
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    fread(solver->u1, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u1);
    fread(solver->u2, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u2);
    fread(solver->u3, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u3);
    fread(solver->p, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_p);

    interp_stag_vel(solver);
}

void IBMSolver_export_results(
    IBMSolver *solver,
    FILE *fp_u1, FILE *fp_u2, FILE *fp_u3, FILE *fp_p
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    fwrite(solver->u1, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u1);
    fwrite(solver->u2, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u2);
    fwrite(solver->u3, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_u3);
    fwrite(solver->p, sizeof(double), (Nx+2)*(Ny+2)*(Nz+2), fp_p);
}

static void alloc_arrays(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    solver->dx = calloc(Nx+2, sizeof(double));
    solver->dy = calloc(Ny+2, sizeof(double));
    solver->dz = calloc(Nz+2, sizeof(double));
    solver->xc = calloc(Nx+2, sizeof(double));
    solver->yc = calloc(Ny+2, sizeof(double));
    solver->zc = calloc(Nz+2, sizeof(double));

    solver->flag = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));

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
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &solver->b);
    HYPRE_IJVectorSetObjectType(solver->b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(solver->b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, Nx*Ny*Nz, &solver->x);
    HYPRE_IJVectorSetObjectType(solver->x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(solver->x);

    solver->vector_rows = calloc(Nx*Ny*Nz, sizeof(int));
    solver->vector_values = calloc(Nx*Ny*Nz, sizeof(double));
    solver->vector_zeros = calloc(Nx*Ny*Nz, sizeof(double));
    solver->vector_res = calloc(Nx*Ny*Nz, sizeof(double));

    for (int i = 0; i < Nx*Ny*Nz; i++) {
        solver->vector_rows[i] = i+1;
    }

    /* Solvers. */
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver->hypre_solver);
    HYPRE_ParCSRBiCGSTABSetLogging(solver->hypre_solver, 1);
    HYPRE_BiCGSTABSetMaxIter(solver->hypre_solver, 1000);
    HYPRE_BiCGSTABSetTol(solver->hypre_solver, 1e-6);
    // HYPRE_BiCGSTABSetPrintLevel(solver_u1, 2);

    HYPRE_BoomerAMGCreate(&solver->precond);
    HYPRE_BoomerAMGSetCoarsenType(solver->precond, 6);
    HYPRE_BoomerAMGSetOldDefault(solver->precond);
    HYPRE_BoomerAMGSetRelaxType(solver->precond, 6);
    HYPRE_BoomerAMGSetNumSweeps(solver->precond, 1);
    HYPRE_BoomerAMGSetTol(solver->precond, 0);
    HYPRE_BoomerAMGSetMaxIter(solver->precond, 1);
    // HYPRE_BoomerAMGSetPrintLevel(precond, 2);

    HYPRE_BiCGSTABSetPrecond(
        solver->hypre_solver,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        solver->precond
    );
}

static HYPRE_IJMatrix create_matrix(IBMSolver *solver, const double3d _lvset, int type) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

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
            if (type != 4) {
                values[0] = 1+ky_N[j]+kx_E[i]+ky_S[j]+kx_W[i]+kz_D[k]+kz_U[k];
            }
            else {
                values[0] = ky_N[j]+kx_E[i]+ky_S[j]+kx_W[i]+kz_D[k]+kz_U[k];
            }
            values[1] = -ky_N[j];
            values[2] = -kx_E[i];
            values[3] = -ky_S[j];
            values[4] = -kx_W[i];
            values[5] = -kz_D[k];
            values[6] = -kz_U[k];

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
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    const double *xc = solver->xc;
    const double *yc = solver->yc;
    const double *zc = solver->zc;

    const double (*lvset)[Ny+2][Nz+2] = _lvset;

    Vector n, m;

    n.x = (lvset[i+1][j][k] - lvset[i-1][j][k]) / (xc[i+1] - xc[i-1]);
    n.y = (lvset[i][j+1][k] - lvset[i][j-1][k]) / (yc[j+1] - yc[j-1]);
    n.z = (lvset[i][j][k+1] - lvset[i][j][k-1]) / (zc[k+1] - zc[k-1]);

    m = Vector_lincom(1, (Vector){xc[i], yc[j], zc[k]}, -2*lvset[i][j][k], n);

    const int im = upper_bound(Nx+2, xc, m.x) - 1;
    const int jm = upper_bound(Ny+2, yc, m.y) - 1;
    const int km = upper_bound(Nz+2, zc, m.z) - 1;

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
