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
    const int Ny_global = solver->Ny_global;
    const int Nz_global = solver->Nz_global;

    double *const buffer = calloc(
        4 * divceil(Nx_global+4, solver->Px) * divceil(Ny_global+4, solver->Py) * divceil(Nz_global+4, solver->Pz),
        sizeof(double)
    );
    int cnt;

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

        float (*const u_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const v_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const w_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const p_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));

        /* Struct containing time and iter. */
        struct time_iter time_iter;

        /* Concatenate extension. */
        snprintf(filename_ext, 100, "%s.nc", filename);

        /* Open file. */
        stat = nc_open(filename_ext, NC_NOWRITE, &ncid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: cannot open file %s\n", filename);
            goto error;
        }

        /* Get variable id. */
        stat = nc_inq_varid(ncid, "time", &time_varid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: varible 'time' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "iter", &iter_varid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: variable 'iter' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "u", &u_varid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: variable 'u' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "v", &v_varid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: variable 'v' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "w", &w_varid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: variable 'w' not found\n");
            goto error;
        }
        stat = nc_inq_varid(ncid, "p", &p_varid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: variable 'p' not found\n");
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

        /* Data to process 0. */
        for (int i = solver->ilower_out; i < solver->iupper_out; i++) {
            for (int j = solver->jlower_out; j < solver->jupper_out; j++) {
                for (int k = solver->klower_out; k < solver->kupper_out; k++) {
                    c3e(solver->u1, i, j, k) = u_value[k+2][j+2][i+2];
                    c3e(solver->u2, i, j, k) = v_value[k+2][j+2][i+2];
                    c3e(solver->u3, i, j, k) = w_value[k+2][j+2][i+2];
                    c3e(solver->p, i, j, k) = p_value[k+2][j+2][i+2];
                }
            }
        }

        /* Set time and iter. */
        time_iter.time = solver->time;
        time_iter.iter = solver->iter;

        /* Send to other processes. */
        for (int rank = 1; rank < solver->num_process; rank++) {
            const int ri = rank / (solver->Py * solver->Pz);
            const int rj = rank % (solver->Py * solver->Pz) / solver->Pz;
            const int rk = rank % solver->Pz;

            const int ilower = ri * (Nx_global+4) / solver->Px - 2;
            const int jlower = rj * (Ny_global+4) / solver->Py - 2;
            const int klower = rk * (Nz_global+4) / solver->Pz - 2;

            const int iupper = (ri+1) * (Nx_global+4) / solver->Px - 2;
            const int jupper = (rj+1) * (Ny_global+4) / solver->Py - 2;
            const int kupper = (rk+1) * (Nz_global+4) / solver->Pz - 2;

            cnt = 0;
            for (int i = ilower; i < iupper; i++) {
                for (int j = jlower; j < jupper; j++) {
                    for (int k = klower; k < kupper; k++) {
                        buffer[cnt++] = u_value[k+2][j+2][i+2];
                        buffer[cnt++] = v_value[k+2][j+2][i+2];
                        buffer[cnt++] = w_value[k+2][j+2][i+2];
                        buffer[cnt++] = p_value[k+2][j+2][i+2];
                    }
                }
            }
            MPI_Send(buffer, 4*(iupper-ilower)*(jupper-jlower)*(kupper-klower), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
            MPI_Send(&time_iter, 1, MPI_DOUBLE_INT, rank, 1, MPI_COMM_WORLD);
        }

        free(u_value);
        free(v_value);
        free(w_value);
        free(p_value);
    }
    else {
        int ifirst, ilast, jfirst, jlast, kfirst, klast;
        struct time_iter time_iter;

        ifirst = solver->ri == 0 ? -2 : 0;
        jfirst = solver->rj == 0 ? -2 : 0;
        kfirst = solver->rk == 0 ? -2 : 0;

        ilast = solver->ri != solver->Px-1 ? Nx : Nx+2;
        jlast = solver->rj != solver->Py-1 ? Ny : Ny+2;
        klast = solver->rk != solver->Pz-1 ? Nz : Nz+2;

        /* Receive from process 0. */
        MPI_Recv(buffer, 4*(ilast-ifirst)*(jlast-jfirst)*(klast-kfirst), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        for (int i = ifirst; i < ilast; i++) {
            for (int j = jfirst; j < jlast; j++) {
                for (int k = kfirst; k < klast; k++) {
                    c3e(solver->u1, i, j, k) = buffer[cnt++];
                    c3e(solver->u2, i, j, k) = buffer[cnt++];
                    c3e(solver->u3, i, j, k) = buffer[cnt++];
                    c3e(solver->p, i, j, k) = buffer[cnt++];
                }
            }
        }

        /* Set time and iter. */
        MPI_Recv(&time_iter, 1, MPI_DOUBLE_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        solver->time = time_iter.time;
        solver->iter = time_iter.iter;
    }

    free(buffer);
    return;

error:
    MPI_Abort(MPI_COMM_WORLD, -1);
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