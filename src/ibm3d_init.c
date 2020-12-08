#include "ibm3d_init.h"

#include "utils.h"
#include <netcdf.h>

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

    solver->iter = 0;
    solver->time = 0;

    for (int i = -2; i < Nx+2; i++) {
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                c3e(solver->u1, i, j, k) = value_u1;
                c3e(solver->u2, i, j, k) = value_u2;
                c3e(solver->u3, i, j, k) = value_u3;
                c3e(solver->p, i, j, k) = value_p;
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

    srand(0);
    FOR_INNER_CELL (i, j, k) {
        u1[i][j][k] += (double)rand() / RAND_MAX * 0.02 - 0.01;
        u2[i][j][k] += (double)rand() / RAND_MAX * 0.02 - 0.01;
        u3[i][j][k] += (double)rand() / RAND_MAX * 0.02 - 0.01;
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

    for (int i = -2; i < Nx+2; i++) {
        for (int j = -2; j < Ny+2; j++) {
            for (int k = -2; k < Nz+2; k++) {
                c3e(solver->u1, i, j, k) = initfunc_u1(c1e(xc, i), c1e(yc, j), c1e(zc, k));
                c3e(solver->u2, i, j, k) = initfunc_u2(c1e(xc, i), c1e(yc, j), c1e(zc, k));
                c3e(solver->u3, i, j, k) = initfunc_u3(c1e(xc, i), c1e(yc, j), c1e(zc, k));
                c3e(solver->p, i, j, k) = initfunc_p(c1e(xc, i), c1e(yc, j), c1e(zc, k));
            }
        }
    }
}