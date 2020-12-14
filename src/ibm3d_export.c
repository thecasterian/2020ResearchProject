#include "ibm3d_export.h"

#include <stdint.h>
#include <string.h>
#include <netcdf.h>
#include <math.h>

#include "utils.h"

static inline int divceil(const int, const int);

/**
 * @brief Exports result in netCDF CF format.
 *
 * @param solver IBMSolver.
 * @param filename File name without an extension.
 */
void IBMSolver_export_result(IBMSolver *solver, const char *filename) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const int Ny_global = solver->Ny_global;
    const int Nz_global = solver->Nz_global;

    const double *const u1 = solver->u1;
    const double *const u2 = solver->u2;
    const double *const u3 = solver->u3;
    const double *const p = solver->p;

    double *const buffer = calloc(
        4 * divceil(Nx_global+4, solver->Px) * divceil(Ny_global+4, solver->Py) * divceil(Nz_global+4, solver->Pz),
        sizeof(double)
    );
    int cnt;

    if (solver->rank == 0) {
        /* File name with extension. */
        char filename_ext[100];

        /* Id of current netCDF file. */
        int ncid;
        /* Id of dimensions. */
        int x_dimid, y_dimid, z_dimid, time_dimid;
        /* Id of variables. */
        int x_varid, y_varid, z_varid, time_varid, iter_varid;
        int u_varid, v_varid, w_varid, p_varid, vort_varid;
        /* Array of dimension ids. */
        int dimids[4];

        /* Value of variables. */
        const double time_value[1] = {solver->time};
        const double *const x_value = solver->xc_global;
        const double *const y_value = solver->yc_global;
        const double *const z_value = solver->zc_global;
        const int iter_value[1] = {solver->iter};

        float (*const u_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const v_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const w_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const p_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+4, sizeof(float [Ny_global+4][Nx_global+4]));
        float (*const vort_value)[Ny_global+4][Nx_global+4] = calloc(Nz_global+2, sizeof(float [Ny_global+4][Nx_global+4]));

        /* Velocity gradients. */
        int iprev, inext, jprev, jnext, kprev, knext;
        float dudy, dudz, dvdx, dvdz, dwdx, dwdy;
        /* Vorticity components. */
        float vortx, vorty, vortz;

        /* netCDF function return value. */
        int stat;

        /* Concatenate extension. */
        snprintf(filename_ext, 100, "%s.nc", filename);

        /* Data from process 0. */
        for (int i = solver->ilower_out; i < solver->iupper_out; i++) {
            for (int j = solver->jlower_out; j < solver->jupper_out; j++) {
                for (int k = solver->klower_out; k < solver->kupper_out; k++) {
                    u_value[k+2][j+2][i+2] = c3e(u1, i, j, k);
                    v_value[k+2][j+2][i+2] = c3e(u2, i, j, k);
                    w_value[k+2][j+2][i+2] = c3e(u3, i, j, k);
                    p_value[k+2][j+2][i+2] = c3e(p, i, j, k);
                }
            }
        }

        /* Receive from other processes. */
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

            MPI_Recv(buffer, 4*(iupper-ilower)*(jupper-jlower)*(kupper-klower), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = ilower; i < iupper; i++) {
                for (int j = jlower; j < jupper; j++) {
                    for (int k = klower; k < kupper; k++) {
                        u_value[k+2][j+2][i+2] = buffer[cnt++];
                        v_value[k+2][j+2][i+2] = buffer[cnt++];
                        w_value[k+2][j+2][i+2] = buffer[cnt++];
                        p_value[k+2][j+2][i+2] = buffer[cnt++];
                    }
                }
            }
        }

        /* Calculate vorticity. */
        for (int k = 0; k < Nz_global+4; k++) {
            for (int j = 0; j < Ny_global+4; j++) {
                for (int i = 0; i < Nx_global+4; i++) {
                    iprev = max(i-1, 0);
                    inext = min(i+1, Nx_global+4);
                    jprev = max(j-1, 0);
                    jnext = min(j+1, Ny_global+4);
                    kprev = max(k-1, 0);
                    knext = min(k+1, Nz_global+4);

                    dudy = (u_value[k][jnext][i] - u_value[k][jprev][i]) / (y_value[jnext] - y_value[jprev]);
                    dudz = (u_value[knext][j][i] - u_value[kprev][j][i]) / (z_value[knext] - z_value[kprev]);
                    dvdx = (v_value[k][j][inext] - v_value[k][j][iprev]) / (x_value[inext] - x_value[iprev]);
                    dvdz = (v_value[knext][j][i] - v_value[kprev][j][i]) / (z_value[knext] - z_value[kprev]);
                    dwdx = (w_value[k][j][inext] - w_value[k][j][iprev]) / (x_value[inext] - x_value[iprev]);
                    dwdy = (w_value[k][jnext][i] - w_value[k][jprev][i]) / (y_value[jnext] - y_value[jprev]);

                    vortx = dwdy - dvdz;
                    vorty = dudz - dwdx;
                    vortz = dvdx - dudy;

                    vort_value[k][j][i] = sqrt(vortx*vortx + vorty*vorty + vortz*vortz);
                }
            }
        }

        fprintf(stderr, "%s\n", filename_ext);

        /* Create file. */
        stat = nc_create(filename_ext, NC_CLOBBER, &ncid);
        fprintf(stderr, "hi\n");
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: cannot open file %s\n", filename_ext);
            goto error;
        }

        /* Define dimensions. */
        nc_def_dim(ncid, "time", 1, &time_dimid);
        nc_def_dim(ncid, "x", Nx_global+4, &x_dimid);
        nc_def_dim(ncid, "y", Ny_global+4, &y_dimid);
        nc_def_dim(ncid, "z", Nz_global+4, &z_dimid);

        /* Define variables. */
        nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid);
        nc_put_att_text(ncid, time_varid, "axis", 1, "T");
        nc_put_att_text(ncid, time_varid, "units", 7, "seconds");

        nc_def_var(ncid, "x", NC_DOUBLE, 1, &x_dimid, &x_varid);
        nc_put_att_text(ncid, x_varid, "axis", 1, "X");
        nc_put_att_text(ncid, x_varid, "units", 6, "meters");

        nc_def_var(ncid, "y", NC_DOUBLE, 1, &y_dimid, &y_varid);
        nc_put_att_text(ncid, y_varid, "axis", 1, "Y");
        nc_put_att_text(ncid, y_varid, "units", 6, "meters");

        nc_def_var(ncid, "z", NC_DOUBLE, 1, &z_dimid, &z_varid);
        nc_put_att_text(ncid, z_varid, "axis", 1, "Z");
        nc_put_att_text(ncid, z_varid, "units", 6, "meters");

        nc_def_var(ncid, "iter", NC_INT, 0, NULL, &iter_varid);

        dimids[0] = time_dimid;
        dimids[1] = z_dimid;
        dimids[2] = y_dimid;
        dimids[3] = x_dimid;

        nc_def_var(ncid, "u", NC_FLOAT, 4, dimids, &u_varid);
        nc_put_att_text(ncid, u_varid, "units", 3, "m/s");

        nc_def_var(ncid, "v", NC_FLOAT, 4, dimids, &v_varid);
        nc_put_att_text(ncid, v_varid, "units", 3, "m/s");

        nc_def_var(ncid, "w", NC_FLOAT, 4, dimids, &w_varid);
        nc_put_att_text(ncid, w_varid, "units", 3, "m/s");

        nc_def_var(ncid, "p", NC_FLOAT, 4, dimids, &p_varid);
        nc_put_att_text(ncid, p_varid, "units", 3, "m/s");

        nc_def_var(ncid, "vorticity", NC_FLOAT, 4, dimids, &vort_varid);
        nc_put_att_text(ncid, vort_varid, "units", 3, "1/s");

        /* End of definitions. */
        nc_enddef(ncid);

        /* Write values. */
        nc_put_var_double(ncid, time_varid, time_value);
        nc_put_var_double(ncid, x_varid, x_value);
        nc_put_var_double(ncid, y_varid, y_value);
        nc_put_var_double(ncid, z_varid, z_value);
        nc_put_var_int(ncid, iter_varid, iter_value);

        nc_put_var_float(ncid, u_varid, (float *)u_value);
        nc_put_var_float(ncid, v_varid, (float *)v_value);
        nc_put_var_float(ncid, w_varid, (float *)w_value);
        nc_put_var_float(ncid, p_varid, (float *)p_value);
        nc_put_var_float(ncid, vort_varid, (float *)vort_value);

        /* Close file. */
        nc_close(ncid);

error:
        free(u_value);
        free(v_value);
        free(w_value);
        free(p_value);
        free(vort_value);
    }
    else {
        int ifirst, ilast, jfirst, jlast, kfirst, klast;

        ifirst = solver->ri == 0 ? -2 : 0;
        jfirst = solver->rj == 0 ? -2 : 0;
        kfirst = solver->rk == 0 ? -2 : 0;

        ilast = solver->ri != solver->Px-1 ? Nx : Nx+2;
        jlast = solver->rj != solver->Py-1 ? Ny : Ny+2;
        klast = solver->rk != solver->Pz-1 ? Nz : Nz+2;

        /* Send to process 0. */
        cnt = 0;
        for (int i = ifirst; i < ilast; i++) {
            for (int j = jfirst; j < jlast; j++) {
                for (int k = kfirst; k < klast; k++) {
                    buffer[cnt++] = c3e(u1, i, j, k);
                    buffer[cnt++] = c3e(u2, i, j, k);
                    buffer[cnt++] = c3e(u3, i, j, k);
                    buffer[cnt++] = c3e(p, i, j, k);
                }
            }
        }
        MPI_Send(buffer, 4*(ilast-ifirst)*(jlast-jfirst)*(klast-kfirst), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    free(buffer);
}

/**
 * @brief Export level set function and flag in netCDF CF format for debug
 *        purpose.
 *
 * @param solver IBMSolver.
 * @param filename File name without an extension.
 */
void IBMSolver_export_lvset_flag(IBMSolver *solver, const char *filename) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;
    const int Ny_global = solver->Ny_global;
    const int Nz_global = solver->Nz_global;

    const double *const lvset = solver->lvset;
    const int *const flag = solver->flag;

    double *const buffer_lvset = calloc(
        divceil(Nx_global+4, solver->Px) * divceil(Ny_global+4, solver->Py) * divceil(Nz_global+4, solver->Pz),
        sizeof(double)
    );
    int *const buffer_flag = calloc(
        divceil(Nx_global+4, solver->Px) * divceil(Ny_global+4, solver->Py) * divceil(Nz_global+4, solver->Pz),
        sizeof(int)
    );
    int cnt;

    if (solver->rank == 0) {
        /* File name with extension. */
        char filename_ext[100];

        /* Id of current netCDF file. */
        int ncid;
        /* Id of dimensions. */
        int x_dimid, y_dimid, z_dimid;
        /* Id of variables. */
        int x_varid, y_varid, z_varid, lvset_varid, flag_varid;
        /* Array of dimension ids. */
        int dimids[3];

        /* Value of variables. */
        const double *const x_value = solver->xc_global;
        const double *const y_value = solver->yc_global;
        const double *const z_value = solver->zc_global;

        float (*const lvset_value)[Ny_global+2][Nx_global+2] = calloc(Nz_global+2, sizeof(float [Ny_global+2][Nx_global+2]));
        int (*const flag_value)[Ny_global+2][Nx_global+2] = calloc(Nz_global+2, sizeof(signed char [Ny_global+2][Nx_global+2]));

        /* netCDF function return value. */
        int stat;

        /* Concatenate extension. */
        snprintf(filename_ext, 100, "%s.nc", filename);

        /* Data from process 0. */
        for (int i = solver->ilower_out; i < solver->iupper_out; i++) {
            for (int j = solver->jlower_out; j < solver->jupper_out; j++) {
                for (int k = solver->klower_out; k < solver->kupper_out; k++) {
                    lvset_value[k+2][j+2][i+2] = c3e(lvset, i, j, k);
                    flag_value[k+2][j+2][i+2] = c3e(flag, i, j, k);
                }
            }
        }

        /* Receive from other processes. */
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

            MPI_Recv(buffer_lvset, (iupper-ilower)*(jupper-jlower)*(kupper-klower), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = ilower; i < iupper; i++) {
                for (int j = jlower; j < jupper; j++) {
                    for (int k = klower; k < kupper; k++) {
                        lvset_value[k+2][j+2][i+2] = buffer_lvset[cnt++];
                    }
                }
            }
            MPI_Recv(buffer_flag, (iupper-ilower)*(jupper-jlower)*(kupper-klower), MPI_INT, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = ilower; i < iupper; i++) {
                for (int j = jlower; j < jupper; j++) {
                    for (int k = klower; k < kupper; k++) {
                        flag_value[k+2][j+2][i+2] = buffer_flag[cnt++];
                    }
                }
            }
        }

        fprintf(stderr, "%s\n", filename_ext);

        /* Create file. */
        stat = nc_create(filename_ext, NC_CLOBBER, &ncid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: cannot open file %s\n", filename_ext);
            goto error;
        }

        /* Define dimensions. */
        nc_def_dim(ncid, "x", Nx_global+4, &x_dimid);
        nc_def_dim(ncid, "y", Ny_global+4, &y_dimid);
        nc_def_dim(ncid, "z", Nz_global+4, &z_dimid);

        /* Define variables. */
        nc_def_var(ncid, "x", NC_DOUBLE, 1, &x_dimid, &x_varid);
        nc_put_att_text(ncid, x_varid, "axis", 1, "X");
        nc_put_att_text(ncid, x_varid, "units", 6, "meters");

        nc_def_var(ncid, "y", NC_DOUBLE, 1, &y_dimid, &y_varid);
        nc_put_att_text(ncid, y_varid, "axis", 1, "Y");
        nc_put_att_text(ncid, y_varid, "units", 6, "meters");

        nc_def_var(ncid, "z", NC_DOUBLE, 1, &z_dimid, &z_varid);
        nc_put_att_text(ncid, z_varid, "axis", 1, "Z");
        nc_put_att_text(ncid, z_varid, "units", 6, "meters");

        dimids[0] = z_dimid;
        dimids[1] = y_dimid;
        dimids[2] = x_dimid;

        nc_def_var(ncid, "lvset", NC_FLOAT, 3, dimids, &lvset_varid);
        nc_def_var(ncid, "flag", NC_INT, 3, dimids, &flag_varid);

        /* End of definitions. */
        nc_enddef(ncid);

        /* Write values. */
        nc_put_var_double(ncid, x_varid, x_value);
        nc_put_var_double(ncid, y_varid, y_value);
        nc_put_var_double(ncid, z_varid, z_value);

        nc_put_var_float(ncid, lvset_varid, (float *)lvset_value);
        nc_put_var_int(ncid, flag_varid, (int *)flag_value);

error:
        free(lvset_value);
        free(flag_value);
    }
    else {
        int ifirst, ilast, jfirst, jlast, kfirst, klast;

        ifirst = solver->ri == 0 ? -2 : 0;
        jfirst = solver->rj == 0 ? -2 : 0;
        kfirst = solver->rk == 0 ? -2 : 0;

        ilast = solver->ri != solver->Px-1 ? Nx : Nx+2;
        jlast = solver->rj != solver->Py-1 ? Ny : Ny+2;
        klast = solver->rk != solver->Pz-1 ? Nz : Nz+2;

        /* Send to process 0. */
        cnt = 0;
        for (int i = ifirst; i < ilast; i++) {
            for (int j = jfirst; j < jlast; j++) {
                for (int k = kfirst; k < klast; k++) {
                    buffer_lvset[cnt++] = c3e(lvset, i, j, k);
                }
            }
        }
        MPI_Send(buffer_lvset, (ilast-ifirst)*(jlast-jfirst)*(klast-kfirst), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        cnt = 0;
                for (int i = ifirst; i < ilast; i++) {
            for (int j = jfirst; j < jlast; j++) {
                for (int k = kfirst; k < klast; k++) {
                    buffer_flag[cnt++] = c3e(flag, i, j, k);
                }
            }
        }
        MPI_Send(buffer_flag, (ilast-ifirst)*(jlast-jfirst)*(klast-kfirst), MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    free(buffer_lvset);
    free(buffer_flag);
}

static inline int divceil(const int a, const int b) {
    return (a + b - 1) / b;
}
