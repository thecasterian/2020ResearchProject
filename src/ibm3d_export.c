#include "ibm3d_export.h"

#include <stdint.h>
#include <string.h>
#include <netcdf.h>
#include <math.h>

#include "utils.h"

void IBMSolver_export_result(IBMSolver *solver, const char *filename) {
    const int Nx = solver->Nx;
    const int Ny = solver->Ny;
    const int Nz = solver->Nz;
    const int Nx_global = solver->Nx_global;

    const double (*const u1)[Ny+2][Nz+2] = solver->u1;
    const double (*const u2)[Ny+2][Nz+2] = solver->u2;
    const double (*const u3)[Ny+2][Nz+2] = solver->u3;
    const double (*const p)[Ny+2][Nz+2] = solver->p;

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
        double time_value[1] = {solver->time};
        double *const x_value = solver->xc_global;
        double *const y_value = solver->yc;
        double *const z_value = solver->zc;
        int iter_value[1] = {solver->iter};

        float (*const u_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const v_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const w_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const p_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));
        float (*const vort_value)[Ny+2][Nx_global+2] = calloc(Nz+2, sizeof(float [Ny+2][Nx_global+2]));

        double (*const u1_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const u2_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const u3_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));
        double (*const p_global)[Ny+2][Nz+2] = calloc(Nx_global+2, sizeof(double [Ny+2][Nz+2]));

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

        /* Convert row-major order to column-major order. */
        for (int i = 0; i <= Nx_global+1; i++) {
            for (int j = 0; j <= Ny+1; j++) {
                for (int k = 0; k <= Nz+1; k++) {
                    u_value[k][j][i] = u1_global[i][j][k];
                    v_value[k][j][i] = u2_global[i][j][k];
                    w_value[k][j][i] = u3_global[i][j][k];
                    p_value[k][j][i] = p_global[i][j][k];
                }
            }
        }

        free(u1_global);
        free(u2_global);
        free(u3_global);
        free(p_global);

        /* Calculate vorticity. */
        for (int k = 0; k <= Nz+1; k++) {
            for (int j = 0; j <= Ny+1; j++) {
                for (int i = 0; i <= Nx_global+1; i++) {
                    iprev = max(i-1, 0);
                    inext = min(i+1, Nx_global+1);
                    jprev = max(j-1, 0);
                    jnext = min(j+1, Ny+1);
                    kprev = max(k-1, 0);
                    knext = min(k+1, Nz+1);

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

        /* Create file. */
        stat = nc_create(filename_ext, NC_CLOBBER, &ncid);
        if (stat != NC_NOERR) {
            fprintf(stderr, "error: cannot open file %s\n", filename_ext);
            goto error;
        }

        /* Define dimensions. */
        nc_def_dim(ncid, "time", 1, &time_dimid);
        nc_def_dim(ncid, "x", Nx_global+2, &x_dimid);
        nc_def_dim(ncid, "y", Ny+2, &y_dimid);
        nc_def_dim(ncid, "z", Nz+2, &z_dimid);

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
