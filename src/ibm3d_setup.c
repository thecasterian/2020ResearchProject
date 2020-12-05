#include "ibm3d_setup.h"

#include "geo3d.h"
#include "utils.h"
#include "math.h"

#include <string.h>
#include <stdarg.h>
#include <netcdf.h>

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
    solver->vel_2nd_halo_lower = solver->vel_2nd_halo_upper = NULL;
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

/**
 * @brief Sets grid informations of the calculation domain.
 *
 * @param solver IBMSolver.
 * @param Nx Number of cells in x-direction.
 * @param Ny Number of cells in y-direction.
 * @param Nz Number of cells in z-direction.
 * @param xf Array of x-coordinates of cell faces in increasing order.
 * @param yf Array of y-coordinates of cell faces in increasing order.
 * @param zf Array of z-coordinates of cell faces in increasing order.
 *
 * @remark Length of \p xf, \p yf, and \p zf must be Nx+1, Ny+1, and Nz+1,
 *         respectively.
 */
void IBMSolver_set_grid(
    IBMSolver *solver,
    const int Nx, const int Ny, const int Nz,
    const double *restrict xf,
    const double *restrict yf,
    const double *restrict zf
) {
    solver->Nx_global = Nx;
    solver->Ny = Ny;
    solver->Nz = Nz;

    solver->ilower = solver->rank * Nx / solver->num_process + 1;
    solver->iupper = (solver->rank+1) * Nx / solver->num_process;
    solver->Nx = solver->iupper - solver->ilower + 1;

    /* Allocate arrays. */
    alloc_arrays(solver);

    /* Cell widths and centroid coordinates. */
    for (int i = 1; i <= solver->Nx_global; i++) {
        solver->dx_global[i] = xf[i] - xf[i-1];
        solver->xc_global[i] = (xf[i] + xf[i-1]) / 2;
    }

    for (int i = 1; i <= solver->Nx; i++) {
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

    /* Min and max coordinates. */
    solver->xmin = xf[0];
    solver->xmax = xf[Nx];
    solver->ymin = yf[0];
    solver->ymax = yf[Ny];
    solver->zmin = zf[0];
    solver->zmax = zf[Nz];
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
            isperiodic(solver->bc[0].type) != isperiodic(solver->bc[2].type)
            || isperiodic(solver->bc[1].type) != isperiodic(solver->bc[3].type)
            || isperiodic(solver->bc[4].type) != isperiodic(solver->bc[5].type)
        )
    ) {
        printf("Inconsistent periodic boundary condition\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Ghost cells */
    dx_global[0] = dx_global[1];
    dx_global[Nx_global+1] = dx_global[Nx_global];
    xc_global[0] = xc_global[1] - (dx_global[0] + dx_global[1]) / 2;
    xc_global[Nx_global+1] = xc_global[Nx_global] + (dx_global[Nx_global] + dx_global[Nx_global+1]) / 2;

    dx[0] = dx_global[ilower-1];
    dx[Nx+1] = dx_global[iupper+1];
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];
    dz[0] = dz[1];
    dz[Nz+1] = dz[Nz];

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

/**
 * @brief Initializes flow with constant values.
 *
 * @param solver IBMSolver.
 * @param value_u1 Initial value of x-velocity.
 * @param value_u2 Initial value of y-velocity.
 * @param value_u3 Initial value of z-velocity.
 * @param value_p Initial value of pressure.
 */
void IBMSolver_init_flow_const(
    IBMSolver *solver,
    const double value_u1, const double value_u2, const double value_u3,
    const double value_p
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    solver->iter = 0;
    solver->time = 0;

    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                u1[i][j][k] = value_u1;
                u2[i][j][k] = value_u2;
                u3[i][j][k] = value_u3;
                p[i][j][k] = value_p;
            }
        }
    }
}

/**
 * @brief Initializes flow with data in the given netCDF CF file. Collective.
 *
 * @param solver IBMSolver.
 * @param filename File name without an extension.
 */
void IBMSolver_init_flow_file(
    IBMSolver *solver,
    const char *filename
) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    double (*const u1)[Ny+2][Nz+2] = solver->u1;
    double (*const u2)[Ny+2][Nz+2] = solver->u2;
    double (*const u3)[Ny+2][Nz+2] = solver->u3;
    double (*const p)[Ny+2][Nz+2] = solver->p;

    struct time_iter {double time; int iter;};

    if (solver->rank == 0) {
        /* File name with extension. */
        char filename_ext[100];

        /* Id of current netCDF file. */
        int ncid;
        /* Id of variables. */
        int time_varid, iter_varid, u_varid, v_varid, w_varid, p_varid;

        /* netCDF function return value. */
        int stat;

        float (*const u_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const v_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const w_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const p_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));

        double (*const u1_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const u2_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const u3_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const p_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));

        /* Struct containing time and iter. */
        struct time_iter time_iter;

        /* Concatenate extension. */
        snprintf(filename_ext, 100, "%s.nc", filename);

        /* Open file. */
        stat = nc_open(filename_ext, NC_NOWRITE, &ncid);
        if (stat != NC_NOERR) {
            printf("cannot open file %s\n", filename);
            goto error;
        }

        /* Get variable id. */
        stat = nc_inq_varid(ncid, "time", &time_varid);
        if (stat != NC_NOERR) {
            printf("varible 'time' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "iter", &iter_varid);
        if (stat != NC_NOERR) {
            printf("variable 'iter' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "u", &u_varid);
        if (stat != NC_NOERR) {
            printf("variable 'u' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "v", &v_varid);
        if (stat != NC_NOERR) {
            printf("variable 'v' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "w", &w_varid);
        if (stat != NC_NOERR) {
            printf("variable 'w' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "p", &p_varid);
        if (stat != NC_NOERR) {
            printf("variable 'p' not found\n");
            goto error;
        }

        /* Read values. */
        nc_get_var_double(ncid, time_varid, &solver->time);
        nc_get_var_int(ncid, iter_varid, &solver->iter);
        nc_get_var_float(ncid, u_varid, (float *)u_value);
        nc_get_var_float(ncid, v_varid, (float *)v_value);
        nc_get_var_float(ncid, w_varid, (float *)w_value);
        nc_get_var_float(ncid, p_varid, (float *)p_value);

        /* Close file. */
        nc_close(ncid);

        /* Convert column-major order to row-major order. */
        for (int i = 0; i <= Nx_global+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                for (int k = 0; k <= Nz+1; k++) {
                    u1_global[i][j][k] = u_value[k][j][i];
                    u2_global[i][j][k] = v_value[k][j][i];
                    u3_global[i][j][k] = w_value[k][j][i];
                    p_global[i][j][k] = p_value[k][j][i];
                }
            }
        }

        free(u_value);
        free(v_value);
        free(w_value);
        free(p_value);

        /* Data to process 0. */
        memcpy(u1, u1_global, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
        memcpy(u2, u2_global, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
        memcpy(u3, u3_global, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
        memcpy(p, p_global, sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

        /* Set time and iter. */
        time_iter.time = solver->time;
        time_iter.iter = solver->iter;

        /* Send to other processes. */
        for (int r = 1; r < solver->num_process; r++) {
            const int ilower_r = r * Nx_global / solver->num_process + 1;
            const int iupper_r = (r+1) * Nx_global / solver->num_process;
            const int Nx_r = iupper_r - ilower_r + 1;

            MPI_Send(u1_global[ilower_r-1], (Nx_r+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(u2_global[ilower_r-1], (Nx_r+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
            MPI_Send(u3_global[ilower_r-1], (Nx_r+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 2, MPI_COMM_WORLD);
            MPI_Send(p_global[ilower_r-1], (Nx_r+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, r, 3, MPI_COMM_WORLD);
            MPI_Send(&time_iter, 1, MPI_DOUBLE_INT, r, 4, MPI_COMM_WORLD);
        }

        free(u1_global);
        free(u2_global);
        free(u3_global);
        free(p_global);

        return;

error:
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    else {
        struct time_iter time_iter;

        /* Receive from process 0. */
        MPI_Recv(u1, (Nx+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(u2, (Nx+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(u3, (Nx+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p, (Nx+2)*(Ny+2)*(Nz+2), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&time_iter, 1, MPI_DOUBLE_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Set time and iter. */
        solver->time = time_iter.time;
        solver->iter = time_iter.iter;
    }
}

/**
 * @brief Initializes flow with functions. Each function takes three arguments
 *        x, y, and z in turn and returns the value of u1, u2, u3, or p for each
 *        point.
 *
 * @param solver IBMSolver.
 * @param initfunc_u1 Initialization function for u1.
 * @param initfunc_u2 Initialization function for u2.
 * @param initfunc_u3 Initialization function for u3.
 * @param initfunc_p Initialization function for p.
 */
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

    if (solver->rank != 0) {
        solver->vel_2nd_halo_lower = calloc(3, sizeof(double [Ny+2][Nz+2]));
    }
    if (solver->rank != solver->num_process - 1) {
        solver->vel_2nd_halo_upper = calloc(3, sizeof(double [Ny+2][Nz+2]));
    }

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
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                for (int k = 0; k <= Nz+1; k++) {
                    lvset[i][j][k] = .5;
                    flag[i][j][k] = FLAG_FLUID;
                }
            }
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
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                if (lvset[i][j][k] >= 0) {
                    flag[i][j][k] = FLAG_FLUID;
                }
                else {
                    bool is_ghost_cell = false;
                    for (int l = 0; l < 6; l++) {
                        int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                        if (ni < 0 || ni > Nx+1 || nj < 0 || nj > Ny+1 || nk < 0 || nk > Nz+1) {
                            continue;
                        }
                        is_ghost_cell = is_ghost_cell || (lvset[ni][nj][nk] >= 0);
                    }
                    flag[i][j][k] = is_ghost_cell ? FLAG_GHOST : FLAG_SOLID;
                }
            }
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
        // HYPRE_BoomerAMGSetPrintLevel(solver->precond_p, 1);

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
        int cols[9] = {cur_idx, 0};
        double values[9] = {0};

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
                switch (solver->bc[3].type) {
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
                switch (solver->bc[1].type) {
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
                switch (solver->bc[2].type) {
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
                switch (solver->bc[0].type) {
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
                switch (solver->bc[4].type) {
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
                switch (solver->bc[5].type) {
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

            /* Normalize pressure equation. */
            if (type == 4) {
                p_coeffsum[i][j][k] = values[0];
                for (int l = 1; l < 7; l++) {
                    values[l] /= values[0];
                }
                values[0] = 1;
            }
        }

        /* Ghost cell. */
        else if (flag[i][j][k] == FLAG_GHOST) {
            int interp_idx[8][3];
            double interp_coeff[8];
            double coeffsum = 0;

            values[0] = 1;

            get_interp_info(solver, i, j, k, interp_idx, interp_coeff);

            /* If an interpolation cell index is out of range, ignore it. */
            for (int l = 0; l < 8; l++) {
                if (
                    interp_idx[l][0] < 1 || interp_idx[l][0] > Nx_global
                    || interp_idx[l][1] < 1 || interp_idx[l][1] > Ny
                    || interp_idx[l][2] < 1 || interp_idx[l][2] > Nz
                ) {
                    interp_coeff[l] = 0;
                }
            }

            /* Normalize. */
            for (int l = 0; l < 8; l++) {
                coeffsum += interp_coeff[l];
            }
            for (int l = 0; l < 8; l++) {
                interp_coeff[l] /= coeffsum;
            }

            /* If the mirror point is not interpolated using the ghost cell
               itself. */
            for (int l = 0; l < 8; l++) {
                if (LOCL_CELL_IDX(interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]) == cur_idx) {
                    if (type != 4) {
                        values[0] += interp_coeff[l];
                    }
                    else {
                        values[0] -= interp_coeff[l];
                    }
                    interp_coeff[l] = 0;
                    break;
                }
            }

            for (int l = 0; l < 8; l++) {
                cols[l+1] = LOCL_CELL_IDX(interp_idx[l][0], interp_idx[l][1], interp_idx[l][2]);
                if (type != 4) {
                    values[l+1] = interp_coeff[l];
                }
                else {
                    values[l+1] = -interp_coeff[l];
                }
            }
        }

        /* Solid cell. */
        else if (flag[i][j][k] == FLAG_SOLID) {
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

    HYPRE_IJMatrixAssemble(A);

    return A;
}

static void get_interp_info(
    IBMSolver *solver,
    const int i, const int j, const int k,
    int interp_idx[restrict][3], double interp_coeff[restrict]
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
        interp_idx[l][0] = im + !!(l & 4);
        interp_idx[l][1] = jm + !!(l & 2);
        interp_idx[l][2] = km + !!(l & 1);
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
