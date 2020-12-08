#include "ibm3d_setup.h"

#include "geo3d.h"
#include "utils.h"
#include "math.h"

#include <string.h>
#include <stdarg.h>

/* Index of adjacent cells in 3-d cartesian coordinate. Refer the order shown
   below. */
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
static void calc_cell_idx(IBMSolver *);
static void calc_lvset_flag(IBMSolver *);
static void build_hypre(IBMSolver *);
static HYPRE_IJMatrix create_matrix(IBMSolver *, int);

static void get_interp_info(
    IBMSolver *,
    const int, const int, const int,
    int [restrict][3], double [restrict]
);
static bool isperiodic(IBMSolverBCType);

/**
 * @brief Makes new IBMSolver. It is dynamically allocated, so its memory must
 *        be freed using IBMSolver_destroy().
 *
 * @param num_process Number of all MPI process in MPI_COMM_WORLD.
 * @param rank Rank of current process in MPI_COMM_WORLD.
 *
 * @return New IBMSolver.
 */
IBMSolver *IBMSolver_new(
    const int num_process, const int rank,
    const int Px, const int Py, const int Pz
) {
    IBMSolver *solver = calloc(1, sizeof(IBMSolver));

    if (rank < 0 || rank > num_process) {
        printf("Invalid rank: %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (Px * Py * Pz != num_process && rank == 0) {
        printf("Number of processes does not match\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    solver->num_process = num_process;
    solver->rank = rank;
    solver->Px = Px;
    solver->Py = Py;
    solver->Pz = Pz;
    solver->ri = rank / (Py * Pz);
    solver->rj = rank % (Py * Pz) / Pz;
    solver->rk = rank % Pz;

    solver->dx = solver->dy = solver->dz = NULL;
    solver->xc = solver->yc = solver->zc = NULL;
    solver->dx_global = solver->dy_global = solver->dz_global = NULL;
    solver->xc_global = solver->yc_global = solver->zc_global = NULL;
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

    return solver;
}

/**
 * @brief Destroys IBMSolver, freeing all its memory in use.
 *
 * @param solver IBMSolver to destroy.
 */
void IBMSolver_destroy(IBMSolver *solver) {
    free(solver->dx); free(solver->dy); free(solver->dz);
    free(solver->xc); free(solver->yc); free(solver->zc);
    free(solver->dx_global); free(solver->dy_global); free(solver->dz_global);
    free(solver->xc_global); free(solver->yc_global); free(solver->zc_global);

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

/**
 * @brief Sets grid informations of the calculation domain.
 *
 * @param solver IBMSolver.
 * @param Nx_global Total number of cells in x-direction.
 * @param Ny_global Total number of cells in y-direction.
 * @param Nz_global Total number of cells in z-direction.
 * @param xf Array of x-coordinates of cell faces in increasing order.
 * @param yf Array of y-coordinates of cell faces in increasing order.
 * @param zf Array of z-coordinates of cell faces in increasing order.
 *
 * @remark Length of \p xf, \p yf, and \p zf must be Nx_global+1, Ny_global+1,
 *         and Nz_global+1, respectively.
 */
void IBMSolver_set_grid(
    IBMSolver *solver,
    const int Nx_global, const int Ny_global, const int Nz_global,
    const double *restrict xf,
    const double *restrict yf,
    const double *restrict zf
) {
    solver->Nx_global = Nx_global;
    solver->Ny_global = Ny_global;
    solver->Nz_global = Nz_global;

    solver->ilower_out = solver->ri * (Nx_global+4) / solver->Px - 2;
    solver->jlower_out = solver->rj * (Ny_global+4) / solver->Py - 2;
    solver->klower_out = solver->rk * (Nz_global+4) / solver->Pz - 2;

    solver->iupper_out = (solver->ri+1) * (Nx_global+4) / solver->Px - 2;
    solver->jupper_out = (solver->rj+1) * (Ny_global+4) / solver->Py - 2;
    solver->kupper_out = (solver->rk+1) * (Nz_global+4) / solver->Pz - 2;

    solver->Nx_out = solver->iupper_out - solver->ilower_out;
    solver->Ny_out = solver->jupper_out - solver->jlower_out;
    solver->Nz_out = solver->kupper_out - solver->klower_out;

    solver->ilower = max(solver->ilower_out, 0);
    solver->jlower = max(solver->jlower_out, 0);
    solver->klower = max(solver->klower_out, 0);

    solver->iupper = min(solver->iupper_out, Nx_global);
    solver->jupper = min(solver->jupper_out, Ny_global);
    solver->kupper = min(solver->kupper_out, Nz_global);

    solver->Nx = solver->iupper - solver->ilower;
    solver->Ny = solver->jupper - solver->jlower;
    solver->Nz = solver->kupper - solver->klower;

    solver->idx_first = (solver->ilower_out+2) * (Ny_global+4)*(Nz_global+4)
        + (solver->jlower_out+2) * (solver->Nx_out)*(Nz_global+4)
        + (solver->klower_out+2) * (solver->Nx_out)*(solver->Ny_out);
    solver->idx_last = solver->idx_first
        + solver->Nx_out*solver->Ny_out*solver->Nz_out;

    /* Allocate arrays. */
    alloc_arrays(solver);

    /* Calculate cell indices. */
    calc_cell_idx(solver);

    /* Cell widths and centroid coordinates. */
    for (int i = 0; i < Nx_global; i++) {
        c1e(solver->dx_global, i) = xf[i+1] - xf[i];
        c1e(solver->xc_global, i) = (xf[i+1] + xf[i]) / 2;
    }
    for (int j = 0; j < Ny_global; j++) {
        c1e(solver->dy_global, j) = yf[j+1] - yf[j];
        c1e(solver->yc_global, j) = (yf[j+1] + yf[j]) / 2;
    }
    for (int k = 0; k < Nz_global; k++) {
        c1e(solver->dz_global, k) = zf[k+1] - zf[k];
        c1e(solver->zc_global, k) = (zf[k+1] + zf[k]) / 2;
    }

    for (int i = 0; i < solver->Nx; i++) {
        c1e(solver->dx, i) = c1e(solver->dx_global, i + solver->ilower);
        c1e(solver->xc, i) = c1e(solver->xc_global, i + solver->ilower);
    }
    for (int j = 0; j < solver->Ny; j++) {
        c1e(solver->dy, j) = c1e(solver->dy_global, j + solver->jlower);
        c1e(solver->yc, j) = c1e(solver->yc_global, j + solver->jlower);
    }
    for (int k = 0; k < solver->Nz; k++) {
        c1e(solver->dz, k) = c1e(solver->dz_global, k + solver->klower);
        c1e(solver->zc, k) = c1e(solver->zc_global, k + solver->klower);
    }

    /* Min and max coordinates. */
    solver->xmin = xf[0];
    solver->xmax = xf[Nx_global];
    solver->ymin = yf[0];
    solver->ymax = yf[Ny_global];
    solver->zmin = zf[0];
    solver->zmax = zf[Nz_global];
}

/**
 * @brief Sets solver parameters in IBMSolver such as Reynolds number, delta-t,
 *        etc.
 *
 * @param solver IBMSolver.
 * @param Re Reynolds number.
 * @param dt Delta-t.
 */
void IBMSolver_set_params(IBMSolver *solver, const double Re, const double dt) {
    solver->Re = Re;
    solver->dt = dt;
}

/**
 * @brief Sets boundary condition of 6 cell boundaries in IBMSolver. Multiple
 *        boundary directions can be provided at once using bitwise-or operator
 *        (|), e.g., DIR_WEST | DIR_EAST. Some boundary types require boundary
 *        value informations. It can be provided as a constant or a function.
 *        For the first case, two additional arguments BC_CONST and the constant
 *        value required. For the second case, two additional arguments BC_FUNC
 *        and the function required. The function takes four arguments t, x, y,
 *        and z in turn and returns the boundary value.
 *
 * @param solver IBMSolver.
 * @param direction Direction of boundary.
 * @param type Type of boundary.
 */
void IBMSolver_set_bc(
    IBMSolver *solver,
    IBMSolverDirection direction,
    IBMSolverBCType type,
    ...
) {
    IBMSolverBCValType val_type = BC_CONST;
    double const_value = NAN;
    IBMSolverBCValFunc func = NULL;
    va_list ap;

    va_start(ap, type);
    switch (type) {
        case BC_VELOCITY_INLET:
        case BC_PRESSURE_OUTLET:
        case BC_VELOCITY_PERIODIC:
            val_type = va_arg(ap, IBMSolverBCValType);
            if (val_type == BC_CONST) {
                const_value = va_arg(ap, double);
            }
            else {
                func = va_arg(ap, IBMSolverBCValFunc);
            }
        break;
        default:;
    }
    va_end(ap);

    for (int i = 0; i < 6; i++) {
        if (direction & (1 << i)) {
            solver->bc[i].type = type;
            solver->bc[i].val_type = val_type;
            if (val_type == BC_CONST) {
                solver->bc[i].const_value = const_value;
            }
            else {
                solver->bc[i].func = func;
            }
        }
    }
}

/**
 * @brief Sets obstacle in IBMSolver. If \p poly is NULL, then it sets no
 *        obstacle in the calculation domain.
 *
 * @param solver IBMSolver.
 * @param poly Obstacle in polyhedron format.
 */
void IBMSolver_set_obstacle(IBMSolver *solver, Polyhedron *poly) {
    solver->poly = poly;
}

/**
 * @brief Sets linear system solver algorithm in IBMSolver.
 *
 * @param solver IBMSolver.
 * @param linear_solver_type Linear system solver type.
 * @param precond_type Preconditioner type. Must be PRECOND_NONE if
 *                     \p linear_solver_type is SOLVER_AMG.
 * @param tol Tolerance. The solver terminates when residual reaches to \p tol.
 */
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

/**
 * @brief Sets autosave file name and period. \p solver automatically exports
 *        the result in netCDF CF format every \p period iterations. Actual
 *        name of an autosaved file is \p filename followed by iteration number
 *        and extension. Non-positive value for \p period disables the autosave.
 *        If the autosave is not set manually, \p period is 0, i.e., autosave
 *        is disabled.
 *
 * @param solver IBMSolver.
 * @param filename Autosave file name without iteration nubmer and extension.
 * @param period Autosave period.
 */
void IBMSolver_set_autosave(
    IBMSolver *solver,
    const char *filename,
    int period
) {
    solver->autosave_filename = filename;
    solver->autosave_period = period;
}

/**
 * @brief Assembles IBMSolver. Must be called after all IBMSolver_set_XXX()
 *        functions are called. Collective.
 *
 * @param solver IBMSolver.
 */
void IBMSolver_assemble(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const int Ny_global = solver->Ny_global;
    const int Nz_global = solver->Nz_global;

    double *const dx_global = solver->dx_global;
    double *const dy_global = solver->dy_global;
    double *const dz_global = solver->dz_global;

    double *const xc_global = solver->xc_global;
    double *const yc_global = solver->yc_global;
    double *const zc_global = solver->zc_global;

    double *const dx = solver->dx;
    double *const dy = solver->dy;
    double *const dz = solver->dz;

    double *const xc = solver->xc;
    double *const yc = solver->yc;
    double *const zc = solver->zc;

    if (
        solver->rank == 0
        && (
            isperiodic(solver->bc[0].type) != isperiodic(solver->bc[2].type)
            || isperiodic(solver->bc[1].type) != isperiodic(solver->bc[3].type)
            || isperiodic(solver->bc[4].type) != isperiodic(solver->bc[5].type)
        )
    ) {
        printf("Inconsistent periodic boundary condition\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Ghost cells */
    c1e(dx_global, -1) = isperiodic(solver->bc[3].type)
        ? c1e(dx_global, Nx_global-1)
        : c1e(dx_global, 0);
    c1e(dx_global, -2) = isperiodic(solver->bc[3].type)
        ? c1e(dx_global, Nx_global-2)
        : c1e(dx_global, 1);
    c1e(dx_global, Nx_global) = isperiodic(solver->bc[1].type)
        ? c1e(dx_global, 0)
        : c1e(dx_global, Nx_global-1);
    c1e(dx_global, Nx_global+1) = isperiodic(solver->bc[1].type)
        ? c1e(dx_global, 1)
        : c1e(dx_global, Nx_global-2);

    c1e(dy_global, -1) = isperiodic(solver->bc[2].type)
        ? c1e(dy_global, Ny_global-1)
        : c1e(dy_global, 0);
    c1e(dy_global, -2) = isperiodic(solver->bc[2].type)
        ? c1e(dy_global, Ny_global-2)
        : c1e(dy_global, 1);
    c1e(dy_global, Ny_global) = isperiodic(solver->bc[0].type)
        ? c1e(dy_global, 0)
        : c1e(dy_global, Ny_global-1);
    c1e(dy_global, Ny_global+1) = isperiodic(solver->bc[0].type)
        ? c1e(dy_global, 1)
        : c1e(dy_global, Ny_global-2);

    c1e(dz_global, -1) = isperiodic(solver->bc[4].type)
        ? c1e(dz_global, Nz_global-1)
        : c1e(dz_global, 0);
    c1e(dz_global, -2) = isperiodic(solver->bc[4].type)
        ? c1e(dz_global, Nz_global-2)
        : c1e(dz_global, 1);
    c1e(dz_global, Nz_global) = isperiodic(solver->bc[5].type)
        ? c1e(dz_global, 0)
        : c1e(dz_global, Nz_global-1);
    c1e(dz_global, Nz_global+1) = isperiodic(solver->bc[5].type)
        ? c1e(dz_global, 1)
        : c1e(dz_global, Nz_global-2);

    c1e(xc_global, -1) = c1e(xc_global, 0) - (c1e(dx_global, -1) + c1e(dx_global, 0)) / 2;
    c1e(xc_global, -2) = c1e(xc_global, -1) - (c1e(dx_global, -2) + c1e(dx_global, -1)) / 2;
    c1e(xc_global, Nx_global+1) = c1e(xc_global, Nx_global) + (c1e(dx_global, Nx_global-1) + c1e(dx_global, Nx_global)) / 2;
    c1e(xc_global, Nx_global+2) = c1e(xc_global, Nx_global+1) + (c1e(dx_global, Nx_global) + c1e(dx_global, Nx_global+1)) / 2;

    c1e(yc_global, -1) = c1e(yc_global, 0) - (c1e(dy_global, -1) + c1e(dy_global, 0)) / 2;
    c1e(yc_global, -2) = c1e(yc_global, -1) - (c1e(dy_global, -2) + c1e(dy_global, -1)) / 2;
    c1e(yc_global, Ny_global+1) = c1e(yc_global, Ny_global) + (c1e(dy_global, Ny_global-1) + c1e(dy_global, Ny_global)) / 2;
    c1e(yc_global, Ny_global+2) = c1e(yc_global, Ny_global+1) + (c1e(dy_global, Ny_global) + c1e(dy_global, Ny_global+1)) / 2;

    c1e(zc_global, -1) = c1e(zc_global, 0) - (c1e(dz_global, -1) + c1e(dz_global, 0)) / 2;
    c1e(zc_global, -2) = c1e(zc_global, -1) - (c1e(dz_global, -2) + c1e(dz_global, -1)) / 2;
    c1e(zc_global, Nz_global+1) = c1e(zc_global, Nz_global) + (c1e(dz_global, Nz_global-1) + c1e(dz_global, Nz_global)) / 2;
    c1e(zc_global, Nz_global+2) = c1e(zc_global, Nz_global+1) + (c1e(dz_global, Nz_global) + c1e(dz_global, Nz_global+1)) / 2;

    c1e(dx, -1) = c1e(dx_global, solver->ilower-1);
    c1e(dx, -2) = c1e(dx_global, solver->ilower-2);
    c1e(dx, Nx) = c1e(dx_global, solver->ilower+Nx);
    c1e(dx, Nx+1) = c1e(dx_global, solver->ilower+Nx+1);

    c1e(dy, -1) = c1e(dy_global, solver->jlower-1);
    c1e(dy, -2) = c1e(dy_global, solver->jlower-2);
    c1e(dy, Ny) = c1e(dy_global, solver->jlower+Ny);
    c1e(dy, Ny+1) = c1e(dy_global, solver->jlower+Ny+1);

    c1e(dz, -1) = c1e(dz_global, solver->klower-1);
    c1e(dz, -2) = c1e(dz_global, solver->klower-2);
    c1e(dz, Nz) = c1e(dz_global, solver->klower+Nz);
    c1e(dz, Nz+1) = c1e(dz_global, solver->klower+Nz+1);

    c1e(xc, -1) = c1e(xc_global, solver->ilower-1);
    c1e(xc, -2) = c1e(xc_global, solver->ilower-2);
    c1e(xc, Nx) = c1e(xc_global, solver->ilower+Nx);
    c1e(xc, Nx+1) = c1e(xc_global, solver->ilower+Nx+1);

    c1e(yc, -1) = c1e(yc_global, solver->jlower-1);
    c1e(yc, -2) = c1e(yc_global, solver->jlower-2);
    c1e(yc, Ny) = c1e(yc_global, solver->jlower+Ny);
    c1e(yc, Ny+1) = c1e(yc_global, solver->jlower+Ny+1);

    c1e(zc, -1) = c1e(zc_global, solver->klower-1);
    c1e(zc, -2) = c1e(zc_global, solver->klower-2);
    c1e(zc, Nz) = c1e(zc_global, solver->klower+Nz);
    c1e(zc, Nz+1) = c1e(zc_global, solver->klower+Nz+1);

    /* Calculate second order derivative coefficients */
    for (int i = 0; i < Nx; i++) {
        c1e(solver->kx_W, i) = solver->dt / (2*solver->Re * (c1e(xc, i) - c1e(xc, i-1))*c1e(dx, i));
        c1e(solver->kx_E, i) = solver->dt / (2*solver->Re * (c1e(xc, i+1) - c1e(xc, i))*c1e(dx, i));
    }
    for (int j = 0; j < Ny; j++) {
        c1e(solver->ky_S, j) = solver->dt / (2*solver->Re * (c1e(yc, j) - c1e(yc, j-1))*c1e(dy, j));
        c1e(solver->ky_N, j) = solver->dt / (2*solver->Re * (c1e(yc, j+1) - c1e(yc, j))*c1e(dy, j));
    }
    for (int k = 0; k < Nz; k++) {
        c1e(solver->kz_D, k) = solver->dt / (2*solver->Re * (c1e(zc, k) - c1e(zc, k-1))*c1e(dz, k));
        c1e(solver->kz_U, k) = solver->dt / (2*solver->Re * (c1e(zc, k+1) - c1e(zc, k))*c1e(dz, k));
    }

    /* Calculate level set function and flag. */
    calc_lvset_flag(solver);

    /* Build HYPRE variables. */
    build_hypre(solver);
}

static void alloc_arrays(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const int Ny_global = solver->Ny_global;
    const int Nz_global = solver->Nz_global;

    solver->cell_idx = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(int));
    solver->cell_idx_periodic = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(int));

    solver->dx = calloc(Nx+4, sizeof(double));
    solver->dy = calloc(Ny+4, sizeof(double));
    solver->dz = calloc(Nz+4, sizeof(double));
    solver->xc = calloc(Nx+4, sizeof(double));
    solver->yc = calloc(Ny+4, sizeof(double));
    solver->zc = calloc(Nz+4, sizeof(double));

    solver->dx_global = calloc(Nx_global+4, sizeof(double));
    solver->dy_global = calloc(Ny_global+4, sizeof(double));
    solver->dz_global = calloc(Nz_global+4, sizeof(double));
    solver->xc_global = calloc(Nx_global+4, sizeof(double));
    solver->yc_global = calloc(Ny_global+4, sizeof(double));
    solver->zc_global = calloc(Nz_global+4, sizeof(double));

    solver->flag = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(int));
    solver->lvset = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));

    solver->u1       = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u1_next  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u1_star  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u1_tilde = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u2       = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u2_next  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u2_star  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u2_tilde = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u3       = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u3_next  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u3_star  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->u3_tilde = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));

    solver->U1      = calloc((Nx+1)*(Ny+4)*(Nz+4), sizeof(double));
    solver->U1_next = calloc((Nx+1)*(Ny+4)*(Nz+4), sizeof(double));
    solver->U1_star = calloc((Nx+1)*(Ny+4)*(Nz+4), sizeof(double));
    solver->U2      = calloc((Nx+4)*(Ny+1)*(Nz+4), sizeof(double));
    solver->U2_next = calloc((Nx+4)*(Ny+1)*(Nz+4), sizeof(double));
    solver->U2_star = calloc((Nx+4)*(Ny+1)*(Nz+4), sizeof(double));
    solver->U3      = calloc((Nx+4)*(Ny+4)*(Nz+1), sizeof(double));
    solver->U3_next = calloc((Nx+4)*(Ny+4)*(Nz+1), sizeof(double));
    solver->U3_star = calloc((Nx+4)*(Ny+4)*(Nz+1), sizeof(double));

    solver->p       = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->p_next  = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->p_prime = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));

    solver->N1      = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->N1_prev = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->N2      = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->N2_prev = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->N3      = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));
    solver->N3_prev = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));

    solver->kx_W = calloc(Nx+4, sizeof(double));
    solver->kx_E = calloc(Nx+4, sizeof(double));
    solver->ky_S = calloc(Ny+4, sizeof(double));
    solver->ky_N = calloc(Ny+4, sizeof(double));
    solver->kz_D = calloc(Nz+4, sizeof(double));
    solver->kz_U = calloc(Nz+4, sizeof(double));

    solver->p_coeffsum = calloc((Nx+4)*(Ny+4)*(Nz+4), sizeof(double));

    solver->x_exchg = calloc(2*(Ny+4)*(Nz+4), sizeof(double));
    solver->y_exchg = calloc(2*(Nx+4)*(Nz+4), sizeof(double));
    solver->z_exchg = calloc(2*(Nx+4)*(Ny+4), sizeof(double));
}

static void calc_cell_idx(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    int ifirst, ilast, jfirst, jlast, kfirst, klast;

    int *x_exchg = calloc(2*(Ny+4)*(Nz+4), sizeof(int));
    int *y_exchg = calloc(2*(Nx+4)*(Nz+4), sizeof(int));
    int *z_exchg = calloc(2*(Nx+4)*(Ny+4), sizeof(int));

    int cnt;

    ifirst = solver->ri == 0 ? -2 : 0;
    ilast = solver->ri == solver->Px-1 ? solver->Nx+2 : solver->Nx;
    jfirst = solver->rj == 0 ? -2 : 0;
    jlast = solver->rj == solver->Py-1 ? solver->Ny+2 : solver->Ny;
    kfirst = solver->rk == 0 ? -2 : 0;
    klast = solver->rk == solver->Pz-1 ? solver->Nz+2 : solver->Nz;

    cnt = solver->idx_first;
    for (int i = ifirst; i < ilast; i++) {
        for (int j = jfirst; j < jlast; j++) {
            for (int k = kfirst; k < klast; k++) {
                c3e(solver->cell_idx, i, j, k) = cnt++;
            }
        }
    }

    /* Exchange cell indices. */
    /* X. */
    if (solver->ri != solver->Px-1) {
        cnt = 0;
        for (int i = Nx-2; i <= Nx-1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    x_exchg[cnt++] = c3e(solver->cell_idx, i, j, k);
                }
            }
        }
        MPI_Send(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != 0) {
        MPI_Recv(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i <= -1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx, i, j, k) = x_exchg[cnt++];
                }
            }
        }
        cnt = 0;
        for (int i = 0; i <= 1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    x_exchg[cnt++] = c3e(solver->cell_idx, i, j, k);
                }
            }
        }
        MPI_Send(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != solver->Px-1) {
        MPI_Recv(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = Nx; i <= Nx+1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx, i, j, k) = x_exchg[cnt++];
                }
            }
        }
    }

    /* Y. */
    if (solver->rj != solver->Py-1) {
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = Ny-2; j <= Ny-1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    y_exchg[cnt++] = c3e(solver->cell_idx, i, j, k);
                }
            }
        }
        MPI_Send(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank + solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != 0) {
        MPI_Recv(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank - solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j <= -1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx, i, j, k) = y_exchg[cnt++];
                }
            }
        }
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = 0; j <= 1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    y_exchg[cnt++] = c3e(solver->cell_idx, i, j, k);
                }
            }
        }
        MPI_Send(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank - solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != solver->Py-1) {
        MPI_Recv(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank + solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = Ny; j <= Ny+1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx, i, j, k) = y_exchg[cnt++];
                }
            }
        }
    }

    /* Z. */
    if (solver->rk != solver->Pz-1) {
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = Nz-2; k <= Nz-1; k++) {
                    z_exchg[cnt++] = c3e(solver->cell_idx, i, j, k);
                }
            }
        }
        MPI_Send(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank + 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != 0) {
        MPI_Recv(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k <= -1; k++) {
                    c3e(solver->cell_idx, i, j, k) = z_exchg[cnt++];
                }
            }
        }
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = 0; k <= 1; k++) {
                    z_exchg[cnt++] = c3e(solver->cell_idx, i, j, k);
                }
            }
        }
        MPI_Send(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank - 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != solver->Pz-1) {
        MPI_Recv(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = Nz; k <= Nz+1; k++) {
                    c3e(solver->cell_idx, i, j, k) = z_exchg[cnt++];
                }
            }
        }
    }

    /* Cell index, but periodic. */
    memcpy(solver->cell_idx_periodic, solver->cell_idx, sizeof(int)*(Nx+4)*(Ny+4)*(Nz+4));

    /* X. */
    if (solver->Px == 1) {
        for (int i = -2; i <= -1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx_periodic, i, j, k) = c3e(solver->cell_idx_periodic, i+Nx, j, k);
                }
            }
        }
        for (int i = Nx; i <= Nx+1; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx_periodic, i, j, k) = c3e(solver->cell_idx_periodic, i-Nx, j, k);
                }
            }
        }
    }
    else {
        if (solver->ri == 0) {
            MPI_Recv(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = -2; i <= -1; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        c3e(solver->cell_idx_periodic, i, j, k) = x_exchg[cnt++];
                    }
                }
            }
            cnt = 0;
            for (int i = 0; i <= 1; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        x_exchg[cnt++] = c3e(solver->cell_idx_periodic, i, j, k);
                    }
                }
            }
            MPI_Send(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank + (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
        }
        else if (solver->ri == solver->Px-1) {
            cnt = 0;
            for (int i = Nx-2; i <= Nx-1; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        x_exchg[cnt++] = c3e(solver->cell_idx_periodic, i, j, k);
                    }
                }
            }
            MPI_Send(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
            MPI_Recv(x_exchg, 2*(Ny+4)*(Nz+4), MPI_INT, solver->rank - (solver->Px-1)*solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = Nx; i <= Nx+1; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        c3e(solver->cell_idx_periodic, i, j, k) = x_exchg[cnt++];
                    }
                }
            }
        }
    }

    /* Y. */
    if (solver->Py == 1) {
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j <= -1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx_periodic, i, j, k) = c3e(solver->cell_idx_periodic, i, j+Ny, k);
                }
            }
        }
        for (int i = -2; i < Nx+2; i++) {
            for (int j = Ny; j <= Ny+1; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->cell_idx_periodic, i, j, k) = c3e(solver->cell_idx_periodic, i, j-Ny, k);
                }
            }
        }
    }
    else {
        if (solver->rj == 0) {
            MPI_Recv(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j <= -1; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        c3e(solver->cell_idx_periodic, i, j, k) = y_exchg[cnt++];
                    }
                }
            }
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = 0; j <= 1; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        y_exchg[cnt++] = c3e(solver->cell_idx_periodic, i, j, k);
                    }
                }
            }
            MPI_Send(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank + (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
        }
        else if (solver->rj == solver->Py-1) {
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = Ny-2; j <= Ny-1; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        y_exchg[cnt++] = c3e(solver->cell_idx_periodic, i, j, k);
                    }
                }
            }
            MPI_Send(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD);
            MPI_Recv(y_exchg, 2*(Nx+4)*(Nz+4), MPI_INT, solver->rank - (solver->Py-1)*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = Ny; j <= Ny+1; j++) {
                    for (int k = -2; k < Nz+2; k++) {
                        c3e(solver->cell_idx_periodic, i, j, k) = y_exchg[cnt++];
                    }
                }
            }
        }
    }

    /* Z. */
    if (solver->Pz == 1) {
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k <= -1; k++) {
                    c3e(solver->cell_idx_periodic, i, j, k) = c3e(solver->cell_idx_periodic, i, j, k+Nz);
                }
            }
        }
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = Nz; k <= Nz+1; k++) {
                    c3e(solver->cell_idx_periodic, i, j, k) = c3e(solver->cell_idx_periodic, i, j, k-Nz);
                }
            }
        }
    }
    else {
        if (solver->rk == 0) {
            MPI_Recv(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = -2; k <= -1; k++) {
                        c3e(solver->cell_idx_periodic, i, j, k) = z_exchg[cnt++];
                    }
                }
            }
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = 0; k <= 1; k++) {
                        z_exchg[cnt++] = c3e(solver->cell_idx_periodic, i, j, k);
                    }
                }
            }
            MPI_Send(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank + (solver->Pz-1), 0, MPI_COMM_WORLD);
        }
        else if (solver->rk == solver->Pz-1) {
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = Nz-2; k <= Nz-1; k++) {
                        z_exchg[cnt++] = c3e(solver->cell_idx_periodic, i, j, k);
                    }
                }
            }
            MPI_Send(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD);
            MPI_Recv(z_exchg, 2*(Nx+4)*(Ny+4), MPI_INT, solver->rank - (solver->Pz-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = -2; i < Nx+2; i++) {
                for (int j = -2; j < Ny+2; j++) {
                    for (int k = Nz; k <= Nz+1; k++) {
                        c3e(solver->cell_idx_periodic, i, j, k) = z_exchg[cnt++];
                    }
                }
            }
        }
    }

    free(x_exchg);
    free(y_exchg);
    free(z_exchg);
}

static void calc_lvset_flag(IBMSolver *solver) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    int *x_exchg = calloc((Ny+4)*(Nz+4), sizeof(int));
    int *y_exchg = calloc((Nx+4)*(Nz+4), sizeof(int));
    int *z_exchg = calloc((Nx+4)*(Ny+4), sizeof(int));

    int cnt;

    /* No obstacle: every cell is fluid cell. */
    if (solver->poly == NULL) {
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                for (int k = -2; k < Nz+2; k++) {
                    c3e(solver->lvset, i, j, k) = .5;
                    c3e(solver->flag, i, j, k) = FLAG_FLUID;
                }
            }
        }
        return;
    }

    /* Calculate level set function. */
    Polyhedron_cpt(
        solver->poly,
        Nx+4, Ny+4, Nz+4,
        solver->xc, solver->yc, solver->zc,
        (double (*)[Ny+4][Nz+4])solver->lvset, .5
    );

    /* Calculate flag.
       * Level set function is positive or zero.  => fluid cell
       * Level set function if negative and at
         least one adjacent cell is fluid cell.   => ghost cell
       * Otherwise.                               => solid cell */
    for (int i = -2; i < Nx+2; i++) {
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                if (c3e(solver->lvset, i, j, k) >= 0) {
                    c3e(solver->flag, i, j, k) = FLAG_FLUID;
                }
                else {
                    bool is_ghost_cell = false;
                    for (int l = 0; l < 6; l++) {
                        int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                        if (ni < -2 || ni > Nx+2 || nj < -2 || nj > Ny+2 || nk < -2 || nk > Nz+2) {
                            continue;
                        }
                        is_ghost_cell = is_ghost_cell || c3e(solver->lvset, ni, nj, nk) >= 0;
                    }
                    c3e(solver->flag, i, j, k) = is_ghost_cell ? FLAG_GHOST : FLAG_SOLID;
                }
            }
        }
    }

    /* Exchange flag between the adjacent processes. */
    if (solver->ri != solver->Px-1) {
        cnt = 0;
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                x_exchg[cnt++] = c3e(solver->flag, Nx-2, j, k);
            }
        }
        MPI_Send(x_exchg, (Ny+4)*(Nz+4), MPI_INT, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != 0) {
        MPI_Recv(x_exchg, (Ny+4)*(Nz+4), MPI_INT, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                c3e(solver->flag, -2, j, k) = x_exchg[cnt++];
            }
        }
        cnt = 0;
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                x_exchg[cnt++] = c3e(solver->flag, 1, j, k);
            }
        }
        MPI_Send(x_exchg, (Ny+4)*(Nz+4), MPI_INT, solver->rank - solver->Py*solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->ri != solver->Px-1) {
        MPI_Recv(x_exchg, (Ny+4)*(Nz+4), MPI_INT, solver->rank + solver->Py*solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                c3e(solver->flag, Nx+1, j, k) = x_exchg[cnt++];
            }
        }
    }

    if (solver->rj != solver->Py-1) {
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int k = -2; k < Nz+2; k++) {
                y_exchg[cnt++] = c3e(solver->flag, i, Ny-2, k);
            }
        }
        MPI_Send(y_exchg, (Nx+4)*(Nz+4), MPI_INT, solver->rank + solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != 0) {
        MPI_Recv(y_exchg, (Nx+4)*(Nz+4), MPI_INT, solver->rank - solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int k = -2; k < Nz+2; k++) {
                c3e(solver->flag, i, -2, k) = y_exchg[cnt++];
            }
        }
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int k = -2; k < Nz+2; k++) {
                y_exchg[cnt++] = c3e(solver->flag, i, 1, k);
            }
        }
        MPI_Send(y_exchg, (Nx+4)*(Nz+4), MPI_INT, solver->rank - solver->Pz, 0, MPI_COMM_WORLD);
    }
    if (solver->rj != solver->Py-1) {
        MPI_Recv(y_exchg, (Nx+4)*(Nz+4), MPI_INT, solver->rank + solver->Pz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int k = -2; k < Nz+2; k++) {
                c3e(solver->flag, i, Nx+1, k) = y_exchg[cnt++];
            }
        }
    }

    if (solver->rk != solver->Pz-1) {
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                z_exchg[cnt++] = c3e(solver->flag, i, j, Nz-2);
            }
        }
        MPI_Send(z_exchg, (Nx+4)*(Ny+4), MPI_INT, solver->rank + 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != 0) {
        MPI_Recv(z_exchg, (Nx+4)*(Ny+4), MPI_INT, solver->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                c3e(solver->flag, i, j, -2) = z_exchg[cnt++];
            }
        }
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                z_exchg[cnt++] = c3e(solver->flag, i, j, 1);
            }
        }
        MPI_Send(z_exchg, (Nx+4)*(Ny+4), MPI_INT, solver->rank - 1, 0, MPI_COMM_WORLD);
    }
    if (solver->rk != solver->Pz-1) {
        MPI_Recv(z_exchg, (Nx+4)*(Ny+4), MPI_INT, solver->rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = -2; i < Nx+2; i++) {
            for (int j = -2; j < Ny+2; j++) {
                c3e(solver->flag, i, j, Nz+1) = z_exchg[cnt++];
            }
        }
    }

    free(x_exchg);
    free(y_exchg);
    free(z_exchg);
}

static void build_hypre(IBMSolver *solver) {
    const int Nx_out = solver->Nx_out;
    const int Ny_out = solver->Ny_out;
    const int Nz_out = solver->Nz_out;

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
        solver->idx_first,
        solver->idx_last-1,
        &solver->b
    );
    HYPRE_IJVectorSetObjectType(solver->b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(solver->b);

    HYPRE_IJVectorCreate(
        MPI_COMM_WORLD,
        solver->idx_first,
        solver->idx_last-1,
        &solver->x
    );
    HYPRE_IJVectorSetObjectType(solver->x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(solver->x);

    solver->vector_rows = calloc(Nx_out*Ny_out*Nz_out, sizeof(int));
    solver->vector_values = calloc(Nx_out*Ny_out*Nz_out, sizeof(double));
    solver->vector_zeros = calloc(Nx_out*Ny_out*Nz_out, sizeof(double));
    solver->vector_res = calloc(Nx_out*Ny_out*Nz_out, sizeof(double));

    for (int i = 0; i < Nx_out*Ny_out*Nz_out; i++) {
        solver->vector_rows[i] = solver->idx_first + i;
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

    HYPRE_IJVectorSetValues(solver->b, Nx_out*Ny_out*Nz_out, solver->vector_rows, solver->vector_zeros);
    HYPRE_IJVectorSetValues(solver->x, Nx_out*Ny_out*Nz_out, solver->vector_rows, solver->vector_zeros);

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

    const double *kx_W = solver->kx_W;
    const double *kx_E = solver->kx_E;
    const double *ky_S = solver->ky_S;
    const double *ky_N = solver->ky_N;
    const double *kz_D = solver->kz_D;
    const double *kz_U = solver->kz_U;

    int ifirst, ilast, jfirst, jlast;

    HYPRE_IJMatrix A;

    int ncols;
    int cur_idx;
    int cols[9];
    double values[9];
    double a, b;

    ifirst = solver->ri == 0 ? -2 : 0;
    ilast = solver->ri == solver->Px-1 ? solver->Nx+2 : solver->Nx;
    jfirst = solver->rj == 0 ? -2 : 0;
    jlast = solver->rj == solver->Py-1 ? solver->Ny+2 : solver->Ny;

    HYPRE_IJMatrixCreate(
        MPI_COMM_WORLD,
        solver->idx_first, solver->idx_last-1,
        solver->idx_first, solver->idx_last-1,
        &A
    );
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    /* Inner cells. */
    FOR_INNER_CELL (i, j, k) {
        cur_idx = c3e(solver->cell_idx, i, j, k);
        for (int l = 0; l < 9; l++) {
            cols[l] = 0;
            values[l] = 0;
        }
        cols[0] = cur_idx;

        /* Fluid cell. */
        if (c3e(solver->flag, i, j, k) == FLAG_FLUID) {
            cols[0] = cur_idx;
            for (int l = 0; l < 6; l++) {
                cols[l+1] = c3e(solver->cell_idx, i+adj[l][0], j+adj[l][1], k+adj[l][2]);
            }
            values[1] = -c1e(ky_N, j);
            values[2] = -c1e(kx_E, i);
            values[3] = -c1e(ky_S, j);
            values[4] = -c1e(kx_W, i);
            values[5] = -c1e(kz_D, k);
            values[6] = -c1e(kz_U, k);
            values[0] = type != 4
                ? 1-values[1]-values[2]-values[3]-values[4]-values[5]-values[6]
                : -values[1]-values[2]-values[3]-values[4]-values[5]-values[6];

            if (type == 4) {
                values[0] -= 1;
            }

            /* Normalize pressure equation. */
            if (type == 4) {
                /* Store center coefficient to normalize RHS also. */
                c3e(solver->p_coeffsum, i, j, k) = values[0];
                for (int l = 1; l < 7; l++) {
                    values[l] /= values[0];
                }
                values[0] = 1;
            }
        }

        // /* Ghost cell. */
        // else if (c3e(solver->flag, i, j, k) == FLAG_GHOST) {
        //     int interp_idx[8][3];
        //     double interp_coeff[8];
        //     double coeffsum;

        //     values[0] = 1;

        //     get_interp_info(solver, i, j, k, interp_idx, interp_coeff);

        //     /* If a solid cell is used for interpolation, ignore it. */
        //     for (int l = 0; l < 8; l++) {
        //         if (c3e(solver->flag, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]) == FLAG_SOLID) {
        //             interp_coeff[l] = 0;
        //         }
        //     }

        //     /* Normalize. */
        //     coeffsum = 0;
        //     for (int l = 0; l < 8; l++) {
        //         coeffsum += interp_coeff[l];
        //     }
        //     for (int l = 0; l < 8; l++) {
        //         interp_coeff[l] /= coeffsum;
        //     }

        //     /* If the mirror point is not interpolated using the ghost cell
        //        itself. */
        //     for (int l = 0; l < 8; l++) {
        //         if (c3e(solver->cell_idx, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]) == cur_idx) {
        //             if (type != 4) {
        //                 values[0] += interp_coeff[l];
        //             }
        //             else {
        //                 values[0] -= interp_coeff[l];
        //             }
        //             interp_coeff[l] = 0;
        //             break;
        //         }
        //     }

        //     for (int l = 0; l < 8; l++) {
        //         cols[l+1] = c3e(solver->cell_idx, interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
        //         if (type != 4) {
        //             values[l+1] = interp_coeff[l];
        //         }
        //         else {
        //             values[l+1] = -interp_coeff[l];
        //         }
        //     }
        // }

        // /* Solid cell. */
        // else if (c3e(solver->flag, i, j, k) == FLAG_SOLID) {
        //     values[0] = 1;
        // }

        else {
            values[0] = 1;
        }

        /* Remove zero elements. */
        ncols = 1;
        for (int l = 1; l < 9; l++) {
            if (values[l] != 0) {
                cols[ncols] = cols[l];
                values[ncols] = values[l];
                ncols++;
            }
        }

        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cur_idx, cols, values);
    }

    /* Outer cells - west. */
    if (solver->ri == 0) {
        switch (solver->bc[3].type) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* i = -2 */
                    cols[0] = c3e(solver->cell_idx, -2, j, k);
                    cols[1] = c3e(solver->cell_idx, 1, j, k);
                    a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
                    b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
                    if (type != 4) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* i = -1 */
                    cols[0] = c3e(solver->cell_idx, -1, j, k);
                    cols[1] = c3e(solver->cell_idx, 0, j, k);
                    a = c1e(solver->dx, -1) / 2;
                    b = c1e(solver->dx, 0) / 2;
                    if (type != 4) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    if (type != 4) {
                        ncols = 3;

                        /* i = -2 */
                        cols[0] = c3e(solver->cell_idx, -2, j, k);
                        cols[1] = c3e(solver->cell_idx, -1, j, k);
                        cols[2] = c3e(solver->cell_idx, 0, j, k);
                        a = c1e(solver->xc, -1) - c1e(solver->xc, -2);
                        b = c1e(solver->xc, 0) - c1e(solver->xc, -1);
                        values[0] = b / (a+b);
                        values[1] = -1;
                        values[2] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = -1 */
                        cols[0] = c3e(solver->cell_idx, -1, j, k);
                        cols[1] = c3e(solver->cell_idx, 0, j, k);
                        cols[2] = c3e(solver->cell_idx, 1, j, k);
                        a = c1e(solver->xc, 0) - c1e(solver->xc, -1);
                        b = c1e(solver->xc, 1) - c1e(solver->xc, 0);
                        values[0] = b / (a+b);
                        values[1] = -1;
                        values[2] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        ncols = 2;

                        /* i = -2 */
                        cols[0] = c3e(solver->cell_idx, -2, j, k);
                        cols[1] = c3e(solver->cell_idx, 1, j, k);
                        a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
                        b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = -1 */
                        cols[0] = c3e(solver->cell_idx, -1, j, k);
                        cols[1] = c3e(solver->cell_idx, 0, j, k);
                        a = c1e(solver->dx, -1) / 2;
                        b = c1e(solver->dx, 0) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* i = -2 */
                    cols[0] = c3e(solver->cell_idx, -2, j, k);
                    cols[1] = c3e(solver->cell_idx, 1, j, k);
                    a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
                    b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
                    if (type == 1) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* i = -1 */
                    cols[0] = c3e(solver->cell_idx, -1, j, k);
                    cols[1] = c3e(solver->cell_idx, 0, j, k);
                    a = c1e(solver->dx, -1) / 2;
                    b = c1e(solver->dx, 0) / 2;
                    if (type == 1) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* i = -2 */
                    cols[0] = c3e(solver->cell_idx, -2, j, k);
                    cols[1] = c3e(solver->cell_idx_periodic, -2, j, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* i = -1 */
                    cols[0] = c3e(solver->cell_idx, -1, j, k);
                    cols[1] = c3e(solver->cell_idx_periodic, -1, j, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_VELOCITY_PERIODIC:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    if (type != 4) {
                        /* i = -2 */
                        cols[0] = c3e(solver->cell_idx, -2, j, k);
                        cols[1] = c3e(solver->cell_idx_periodic, -2, j, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = -1 */
                        cols[0] = c3e(solver->cell_idx, -1, j, k);
                        cols[1] = c3e(solver->cell_idx_periodic, -1, j, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        /* i = -2 */
                        cols[0] = c3e(solver->cell_idx, -2, j, k);
                        cols[1] = c3e(solver->cell_idx, 1, j, k);
                        a = c1e(solver->dx, -2) / 2 + c1e(solver->dx, -1);
                        b = c1e(solver->dx, 0) + c1e(solver->dx, 1) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = -1 */
                        cols[0] = c3e(solver->cell_idx, -1, j, k);
                        cols[1] = c3e(solver->cell_idx, 0, j, k);
                        a = c1e(solver->dx, -1) / 2;
                        b = c1e(solver->dx, 0) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Outer cells - east. */
    if (solver->ri == solver->Px-1) {
        switch (solver->bc[1].type) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* i = Nx */
                    cols[0] = c3e(solver->cell_idx, Nx, j, k);
                    cols[1] = c3e(solver->cell_idx, Nx-1, j, k);
                    a = c1e(solver->dx, Nx-1) / 2;
                    b = c1e(solver->dx, Nx) / 2;
                    if (type != 4) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* i = Nx+1 */
                    cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                    cols[1] = c3e(solver->cell_idx, Nx-2, j, k);
                    a = c1e(solver->dx, Nx-2) + c1e(solver->dx, Nx-1) / 2;
                    b = c1e(solver->dx, Nx) / 2 + c1e(solver->dx, Nx+1);
                    if (type != 4) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    if (type != 4) {
                        ncols = 3;

                        /* i = Nx */
                        cols[0] = c3e(solver->cell_idx, Nx, j, k);
                        cols[1] = c3e(solver->cell_idx, Nx-1, j, k);
                        cols[2] = c3e(solver->cell_idx, Nx-2, j, k);
                        a = c1e(solver->xc, Nx-1) - c1e(solver->xc, Nx-2);
                        b = c1e(solver->xc, Nx) - c1e(solver->xc, Nx-1);
                        values[0] = a / (a+b);
                        values[1] = -1;
                        values[2] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = Nx+1 */
                        cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                        cols[1] = c3e(solver->cell_idx, Nx, j, k);
                        cols[2] = c3e(solver->cell_idx, Nx-1, j, k);
                        a = c1e(solver->xc, Nx) - c1e(solver->xc, Nx-1);
                        b = c1e(solver->xc, Nx+1) - c1e(solver->xc, Nx);
                        values[0] = a / (a+b);
                        values[1] = -1;
                        values[2] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        ncols = 2;

                        /* i = Nx */
                        cols[0] = c3e(solver->cell_idx, Nx, j, k);
                        cols[1] = c3e(solver->cell_idx, Nx-1, j, k);
                        a = c1e(solver->dx, Nx-1) / 2;
                        b = c1e(solver->dx, Nx) / 2;
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = Nx+1 */
                        cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                        cols[1] = c3e(solver->cell_idx, Nx-2, j, k);
                        a = c1e(solver->dx, Nx-2) + c1e(solver->dx, Nx-1) / 2;
                        b = c1e(solver->dx, Nx) / 2 + c1e(solver->dx, Nx+1);
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* i = Nx */
                    cols[0] = c3e(solver->cell_idx, Nx, j, k);
                    cols[1] = c3e(solver->cell_idx, Nx-1, j, k);
                    a = c1e(solver->dx, Nx-1) / 2;
                    b = c1e(solver->dx, Nx) / 2;
                    if (type == 1) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* i = Nx+1 */
                    cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                    cols[1] = c3e(solver->cell_idx, Nx-2, j, k);
                    a = c1e(solver->dx, Nx-2) + c1e(solver->dx, Nx-1) / 2;
                    b = c1e(solver->dx, Nx) / 2 + c1e(solver->dx, Nx+1);
                    if (type == 1) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* i = Nx */
                    cols[0] = c3e(solver->cell_idx, Nx, j, k);
                    cols[1] = c3e(solver->cell_idx_periodic, Nx, j, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* i = Nx+1 */
                    cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                    cols[1] = c3e(solver->cell_idx_periodic, Nx+1, j, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_VELOCITY_PERIODIC:
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    if (type != 4) {
                        /* i = Nx */
                        cols[0] = c3e(solver->cell_idx, Nx, j, k);
                        cols[1] = c3e(solver->cell_idx_periodic, Nx, j, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = Nx+1 */
                        cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                        cols[1] = c3e(solver->cell_idx_periodic, Nx+1, j, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        /* i = Nx */
                        cols[0] = c3e(solver->cell_idx, Nx, j, k);
                        cols[1] = c3e(solver->cell_idx, Nx-1, j, k);
                        a = c1e(solver->dx, Nx-1) / 2;
                        b = c1e(solver->dx, Nx) / 2;
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* i = Nx+1 */
                        cols[0] = c3e(solver->cell_idx, Nx+1, j, k);
                        cols[1] = c3e(solver->cell_idx, Nx-2, j, k);
                        a = c1e(solver->dx, Nx-2) + c1e(solver->dx, Nx-1) / 2;
                        b = c1e(solver->dx, Nx) / 2 + c1e(solver->dx, Nx+1);
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Outer cells - south. */
    if (solver->rj == 0) {
        switch (solver->bc[2].type) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* j = -2 */
                    cols[0] = c3e(solver->cell_idx, i, -2, k);
                    cols[1] = c3e(solver->cell_idx, i, 1, k);
                    a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
                    b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
                    if (type != 4) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* j = -1 */
                    cols[0] = c3e(solver->cell_idx, i, -1, k);
                    cols[1] = c3e(solver->cell_idx, i, 0, k);
                    a = c1e(solver->dy, -1) / 2;
                    b = c1e(solver->dy, 0) / 2;
                    if (type != 4) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    if (type != 4) {
                        ncols = 3;

                        /* j = -2 */
                        cols[0] = c3e(solver->cell_idx, i, -2, k);
                        cols[1] = c3e(solver->cell_idx, i, -1, k);
                        cols[2] = c3e(solver->cell_idx, i, 0, k);
                        a = c1e(solver->yc, -1) - c1e(solver->yc, -2);
                        b = c1e(solver->yc, 0) - c1e(solver->yc, -1);
                        values[0] = b / (a+b);
                        values[1] = -1;
                        values[2] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = -1 */
                        cols[0] = c3e(solver->cell_idx, i, -1, k);
                        cols[1] = c3e(solver->cell_idx, i, 0, k);
                        cols[2] = c3e(solver->cell_idx, i, 1, k);
                        a = c1e(solver->yc, 0) - c1e(solver->yc, -1);
                        b = c1e(solver->yc, 1) - c1e(solver->yc, 0);
                        values[0] = b / (a+b);
                        values[1] = -1;
                        values[2] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        ncols = 2;

                        /* j = -2 */
                        cols[0] = c3e(solver->cell_idx, i, -2, k);
                        cols[1] = c3e(solver->cell_idx, i, 1, k);
                        a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
                        b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = -1 */
                        cols[0] = c3e(solver->cell_idx, i, -1, k);
                        cols[1] = c3e(solver->cell_idx, i, 0, k);
                        a = c1e(solver->dy, -1) / 2;
                        b = c1e(solver->dy, 0) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* j = -2 */
                    cols[0] = c3e(solver->cell_idx, i, -2, k);
                    cols[1] = c3e(solver->cell_idx, i, 1, k);
                    a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
                    b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
                    if (type == 2) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* j = -1 */
                    cols[0] = c3e(solver->cell_idx, i, -1, k);
                    cols[1] = c3e(solver->cell_idx, i, 0, k);
                    a = c1e(solver->dy, -1) / 2;
                    b = c1e(solver->dy, 0) / 2;
                    if (type == 2) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* j = -2 */
                    cols[0] = c3e(solver->cell_idx, i, -2, k);
                    cols[1] = c3e(solver->cell_idx_periodic, i, -2, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* j = -1 */
                    cols[0] = c3e(solver->cell_idx, i, -1, k);
                    cols[1] = c3e(solver->cell_idx_periodic, i, -1, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    if (type != 4) {
                        /* j = -2 */
                        cols[0] = c3e(solver->cell_idx, i, -2, k);
                        cols[1] = c3e(solver->cell_idx_periodic, i, -2, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = -1 */
                        cols[0] = c3e(solver->cell_idx, i, -1, k);
                        cols[1] = c3e(solver->cell_idx_periodic, i, -1, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        /* j = -2 */
                        cols[0] = c3e(solver->cell_idx, i, -2, k);
                        cols[1] = c3e(solver->cell_idx, i, 1, k);
                        a = c1e(solver->dy, -2) / 2 + c1e(solver->dy, -1);
                        b = c1e(solver->dy, 0) + c1e(solver->dy, 1) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = -1 */
                        cols[0] = c3e(solver->cell_idx, i, -1, k);
                        cols[1] = c3e(solver->cell_idx, i, 0, k);
                        a = c1e(solver->dy, -1) / 2;
                        b = c1e(solver->dy, 0) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Outer cells - north. */
    if (solver->rj == solver->Py-1) {
        switch (solver->bc[0].type) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* j = Ny */
                    cols[0] = c3e(solver->cell_idx, i, Ny, k);
                    cols[1] = c3e(solver->cell_idx, i, Ny-1, k);
                    a = c1e(solver->dy, Ny-1) / 2;
                    b = c1e(solver->dy, Ny) / 2;
                    if (type != 4) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* j = Ny+1 */
                    cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                    cols[1] = c3e(solver->cell_idx, i, Ny-2, k);
                    a = c1e(solver->dy, Ny-2) + c1e(solver->dy, Ny-1) / 2;
                    b = c1e(solver->dy, Ny) / 2 + c1e(solver->dy, Ny+1);
                    if (type != 4) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    if (type != 4) {
                        ncols = 3;

                        /* j = Ny */
                        cols[0] = c3e(solver->cell_idx, i, Ny, k);
                        cols[1] = c3e(solver->cell_idx, i, Ny-1, k);
                        cols[2] = c3e(solver->cell_idx, i, Ny-2, k);
                        a = c1e(solver->yc, Ny-1) - c1e(solver->yc, Ny-2);
                        b = c1e(solver->yc, Ny) - c1e(solver->yc, Ny-1);
                        values[0] = a / (a+b);
                        values[1] = -1;
                        values[2] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = Ny+1 */
                        cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                        cols[1] = c3e(solver->cell_idx, i, Ny, k);
                        cols[2] = c3e(solver->cell_idx, i, Ny-1, k);
                        a = c1e(solver->yc, Ny) - c1e(solver->yc, Ny-1);
                        b = c1e(solver->yc, Ny+1) - c1e(solver->yc, Ny);
                        values[0] = a / (a+b);
                        values[1] = -1;
                        values[2] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        ncols = 2;

                        /* j = Ny */
                        cols[0] = c3e(solver->cell_idx, i, Ny, k);
                        cols[1] = c3e(solver->cell_idx, i, Ny-1, k);
                        a = c1e(solver->dy, Ny-1) / 2;
                        b = c1e(solver->dy, Ny) / 2;
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = Ny+1 */
                        cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                        cols[1] = c3e(solver->cell_idx, i, Ny-2, k);
                        a = c1e(solver->dy, Ny-2) + c1e(solver->dy, Ny-1) / 2;
                        b = c1e(solver->dy, Ny) / 2 + c1e(solver->dy, Ny+1);
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* j = Ny */
                    cols[0] = c3e(solver->cell_idx, i, Ny, k);
                    cols[1] = c3e(solver->cell_idx, i, Ny-1, k);
                    a = c1e(solver->dy, Ny-1) / 2;
                    b = c1e(solver->dy, Ny) / 2;
                    if (type == 2) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* j = Ny+1 */
                    cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                    cols[1] = c3e(solver->cell_idx, i, Ny-2, k);
                    a = c1e(solver->dy, Ny-2) + c1e(solver->dy, Ny-1) / 2;
                    b = c1e(solver->dy, Ny) / 2 + c1e(solver->dy, Ny+1);
                    if (type == 2) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    /* j = Ny */
                    cols[0] = c3e(solver->cell_idx, i, Ny, k);
                    cols[1] = c3e(solver->cell_idx_periodic, i, Ny, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* j = Ny+1 */
                    cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                    cols[1] = c3e(solver->cell_idx_periodic, i, Ny+1, k);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int k = 0; k < Nz; k++) {
                    ncols = 2;

                    if (type != 4) {
                        /* j = Ny */
                        cols[0] = c3e(solver->cell_idx, i, Ny, k);
                        cols[1] = c3e(solver->cell_idx_periodic, i, Ny, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = Ny+1 */
                        cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                        cols[1] = c3e(solver->cell_idx_periodic, i, Ny+1, k);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        /* j = Ny */
                        cols[0] = c3e(solver->cell_idx, i, Ny, k);
                        cols[1] = c3e(solver->cell_idx, i, Ny-1, k);
                        a = c1e(solver->dy, Ny-1) / 2;
                        b = c1e(solver->dy, Ny) / 2;
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* j = Ny+1 */
                        cols[0] = c3e(solver->cell_idx, i, Ny+1, k);
                        cols[1] = c3e(solver->cell_idx, i, Ny-2, k);
                        a = c1e(solver->dy, Ny-2) + c1e(solver->dy, Ny-1) / 2;
                        b = c1e(solver->dy, Ny) / 2 + c1e(solver->dy, Ny+1);
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Outer cells - down. */
    if (solver->rk == 0) {
        switch (solver->bc[4].type) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    /* k = -2 */
                    cols[0] = c3e(solver->cell_idx, i, j, -2);
                    cols[1] = c3e(solver->cell_idx, i, j, 1);
                    a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
                    b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
                    if (type != 4) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* k = -1 */
                    cols[0] = c3e(solver->cell_idx, i, j, -1);
                    cols[1] = c3e(solver->cell_idx, i, j, 0);
                    a = c1e(solver->dz, -1) / 2;
                    b = c1e(solver->dz, 0) / 2;
                    if (type != 4) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    if (type != 4) {
                        ncols = 3;

                        /* k = -2 */
                        cols[0] = c3e(solver->cell_idx, i, j, -2);
                        cols[1] = c3e(solver->cell_idx, i, j, -1);
                        cols[2] = c3e(solver->cell_idx, i, j, 0);
                        a = c1e(solver->zc, -1) - c1e(solver->zc, -2);
                        b = c1e(solver->zc, 0) - c1e(solver->zc, -1);
                        values[0] = b / (a+b);
                        values[1] = -1;
                        values[2] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = -1 */
                        cols[0] = c3e(solver->cell_idx, i, j, -1);
                        cols[1] = c3e(solver->cell_idx, i, j, 0);
                        cols[2] = c3e(solver->cell_idx, i, j, 1);
                        a = c1e(solver->zc, 0) - c1e(solver->zc, -1);
                        b = c1e(solver->zc, 1) - c1e(solver->zc, 0);
                        values[0] = b / (a+b);
                        values[1] = -1;
                        values[2] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        ncols = 2;

                        /* k = -2 */
                        cols[0] = c3e(solver->cell_idx, i, j, -2);
                        cols[1] = c3e(solver->cell_idx, i, j, 1);
                        a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
                        b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = -1 */
                        cols[0] = c3e(solver->cell_idx, i, j, -1);
                        cols[1] = c3e(solver->cell_idx, i, j, 0);
                        a = c1e(solver->dz, -1) / 2;
                        b = c1e(solver->dz, 0) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    /* k = -2 */
                    cols[0] = c3e(solver->cell_idx, i, j, -2);
                    cols[1] = c3e(solver->cell_idx, i, j, 1);
                    a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
                    b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
                    if (type == 3) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* k = -1 */
                    cols[0] = c3e(solver->cell_idx, i, j, -1);
                    cols[1] = c3e(solver->cell_idx, i, j, 0);
                    a = c1e(solver->dz, -1) / 2;
                    b = c1e(solver->dz, 0) / 2;
                    if (type == 3) {
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    /* k = -2 */
                    cols[0] = c3e(solver->cell_idx, i, j, -2);
                    cols[1] = c3e(solver->cell_idx_periodic, i, j, -2);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* k = -1 */
                    cols[0] = c3e(solver->cell_idx, i, j, -1);
                    cols[1] = c3e(solver->cell_idx_periodic, i, j, -1);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    if (type != 4) {
                        /* k = -2 */
                        cols[0] = c3e(solver->cell_idx, i, j, -2);
                        cols[1] = c3e(solver->cell_idx_periodic, i, j, -2);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = -1 */
                        cols[0] = c3e(solver->cell_idx, i, j, -1);
                        cols[1] = c3e(solver->cell_idx_periodic, i, j, -1);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        /* k = -2 */
                        cols[0] = c3e(solver->cell_idx, i, j, -2);
                        cols[1] = c3e(solver->cell_idx, i, j, 1);
                        a = c1e(solver->dz, -2) / 2 + c1e(solver->dz, -1);
                        b = c1e(solver->dz, 0) + c1e(solver->dz, 1) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = -1 */
                        cols[0] = c3e(solver->cell_idx, i, j, -1);
                        cols[1] = c3e(solver->cell_idx, i, j, 0);
                        a = c1e(solver->dz, -1) / 2;
                        b = c1e(solver->dz, 0) / 2;
                        values[0] = b / (a+b);
                        values[1] = a / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        default:;
        }
    }

    /* Outer cells - up. */
    if (solver->rk == solver->Pz-1) {
        switch (solver->bc[5].type) {
        case BC_VELOCITY_INLET:
        case BC_STATIONARY_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    /* k = Nz */
                    cols[0] = c3e(solver->cell_idx, i, j, Nz);
                    cols[1] = c3e(solver->cell_idx, i, j, Nz-1);
                    a = c1e(solver->dy, Ny-1) / 2;
                    b = c1e(solver->dy, Ny) / 2;
                    if (type != 4) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* k = Nz+1 */
                    cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                    cols[1] = c3e(solver->cell_idx, i, j, Nz-2);
                    a = c1e(solver->dz, Nz-2) + c1e(solver->dz, Nz-1) / 2;
                    b = c1e(solver->dz, Nz) / 2 + c1e(solver->dz, Nz+1);
                    if (type != 4) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_PRESSURE_OUTLET:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    if (type != 4) {
                        ncols = 3;

                        /* k = Nz */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz);
                        cols[1] = c3e(solver->cell_idx, i, j, Nz-1);
                        cols[2] = c3e(solver->cell_idx, i, j, Nz-2);
                        a = c1e(solver->zc, Nz-1) - c1e(solver->zc, Nz-2);
                        b = c1e(solver->zc, Nz) - c1e(solver->zc, Nz-1);
                        values[0] = a / (a+b);
                        values[1] = -1;
                        values[2] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = Nz+1 */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                        cols[1] = c3e(solver->cell_idx, i, j, Nz);
                        cols[2] = c3e(solver->cell_idx, i, j, Nz-1);
                        a = c1e(solver->zc, Nz) - c1e(solver->zc, Nz-1);
                        b = c1e(solver->zc, Nz+1) - c1e(solver->zc, Nz);
                        values[0] = a / (a+b);
                        values[1] = -1;
                        values[2] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        ncols = 2;

                        /* k = Nz */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz);
                        cols[1] = c3e(solver->cell_idx, i, j, Nz-1);
                        a = c1e(solver->dz, Nz-1) / 2;
                        b = c1e(solver->dz, Nz) / 2;
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = Nz+1 */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                        cols[1] = c3e(solver->cell_idx, i, j, Nz-2);
                        a = c1e(solver->dz, Nz-2) + c1e(solver->dz, Nz-1) / 2;
                        b = c1e(solver->dz, Nz) / 2 + c1e(solver->dz, Nz+1);
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        case BC_FREE_SLIP_WALL:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    /* k = Nz */
                    cols[0] = c3e(solver->cell_idx, i, j, Nz);
                    cols[1] = c3e(solver->cell_idx, i, j, Nz-1);
                    a = c1e(solver->dz, Nz-1) / 2;
                    b = c1e(solver->dz, Nz) / 2;
                    if (type == 3) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* k = Nz+1 */
                    cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                    cols[1] = c3e(solver->cell_idx, i, j, Nz-2);
                    a = c1e(solver->dz, Nz-2) + c1e(solver->dz, Nz-1) / 2;
                    b = c1e(solver->dz, Nz) / 2 + c1e(solver->dz, Nz+1);
                    if (type == 3) {
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                    }
                    else {
                        values[0] = 1;
                        values[1] = -1;
                    }
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_ALL_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    /* k = Nz */
                    cols[0] = c3e(solver->cell_idx, i, j, Nz);
                    cols[1] = c3e(solver->cell_idx_periodic, i, j, Nz);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                    /* k = Nz+1 */
                    cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                    cols[1] = c3e(solver->cell_idx_periodic, i, j, Nz+1);
                    values[0] = 1;
                    values[1] = -1;
                    HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                }
            }
            break;
        case BC_VELOCITY_PERIODIC:
            for (int i = ifirst; i < ilast; i++) {
                for (int j = jfirst; j < jlast; j++) {
                    ncols = 2;

                    if (type != 4) {
                        /* k = Nz */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz);
                        cols[1] = c3e(solver->cell_idx_periodic, i, j, Nz);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = Nz+1 */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                        cols[1] = c3e(solver->cell_idx_periodic, i, j, Nz+1);
                        values[0] = 1;
                        values[1] = -1;
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                    else {
                        /* k = Nz */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz);
                        cols[1] = c3e(solver->cell_idx, i, j, Nz-1);
                        a = c1e(solver->dz, Nz-1) / 2;
                        b = c1e(solver->dz, Nz) / 2;
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);

                        /* k = Nz+1 */
                        cols[0] = c3e(solver->cell_idx, i, j, Nz+1);
                        cols[1] = c3e(solver->cell_idx, i, j, Nz-2);
                        a = c1e(solver->dz, Nz-2) + c1e(solver->dz, Nz-1) / 2;
                        b = c1e(solver->dz, Nz) / 2 + c1e(solver->dz, Nz+1);
                        values[0] = a / (a+b);
                        values[1] = b / (a+b);
                        HYPRE_IJMatrixSetValues(A, 1, &ncols, &cols[0], cols, values);
                    }
                }
            }
            break;
        default:;
        }
    }

    HYPRE_IJMatrixAssemble(A);

    return A;
}

static void get_interp_info(
    IBMSolver *solver,
    const int i, const int j, const int k,
    int interp_idx[restrict][3], double interp_coeff[restrict]
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    /* Normal vector. */
    Vector n;
    /* Mirror point. */
    Vector m;

    n.x = (c3e(solver->lvset, i+1, j, k) - c3e(solver->lvset, i-1, j, k))
        / (c1e(solver->xc, i+1) - c1e(solver->xc, i-1));
    n.x = (c3e(solver->lvset, i, j+1, k) - c3e(solver->lvset, i, j-1, k))
        / (c1e(solver->yc, j+1) - c1e(solver->yc, j-1));
    n.x = (c3e(solver->lvset, i, j, k+1) - c3e(solver->lvset, i, j, k-1))
        / (c1e(solver->zc, k+1) - c1e(solver->zc, k-1));

    m = Vector_lincom(
        1, (Vector){c1e(solver->xc, i), c1e(solver->yc, j), c1e(solver->zc, k)},
        -2*c3e(solver->lvset, i, j, k), n
    );

    const int im = upper_bound_double(Nx+4, solver->xc, m.x) - 3;
    const int jm = upper_bound_double(Ny+4, solver->yc, m.y) - 3;
    const int km = upper_bound_double(Nz+4, solver->zc, m.z) - 3;

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
        interp_idx[l][0] = im + !!(l & 4);
        interp_idx[l][1] = jm + !!(l & 2);
        interp_idx[l][2] = km + !!(l & 1);
    }

    const double xl = c1e(solver->xc, im), xu = c1e(solver->xc, im+1);
    const double yl = c1e(solver->yc, jm), yu = c1e(solver->yc, jm+1);
    const double zl = c1e(solver->zc, km), zu = c1e(solver->zc, km+1);
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
