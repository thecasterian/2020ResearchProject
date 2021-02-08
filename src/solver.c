#include "../include/solver.h"

#include <stdio.h>
#include <stdlib.h>
#include "../include/mesh.h"
#include "../include/vars.h"
#include "../include/init.h"
#include "../include/utils.h"
#include "../include/netcdf_io.h"
#include <mpi.h>
#include <netcdf.h>

#ifndef __USE_POSIX199309
#define __USE_POSIX199309
#endif
#include <time.h>

Solver *Solver_Create(PartMesh *mesh, Params *params) {
    Solver *solver;

    solver = calloc(1, sizeof(*solver));

    solver->mesh = mesh;
    solver->params = params;
    solver->vars = FlowVars_Create(mesh);

    return solver;
}

void Solver_Assemble(Solver *solver) {
    PartMesh_CalcTransform(solver->mesh);

}

void Solver_Initialize(Solver *solver, Initializer *init) {
    FlowVars_Initialize(solver->vars, solver->mesh, init);
}

void Solver_Iterate(Solver *solver, int ntimesteps, bool verbose) {
    struct timespec t_start, t_end;
    long elapsed_time, hour, min, sec;

    int i = 0;
    double start_time = solver->time;

    FlowVars_UpdateOuter(solver->vars, solver->mesh);
    FlowVars_AdjExchg(solver->vars, solver->mesh);
}

void Solver_ExportNetCDF(Solver *solver, NetCDFWriter *writer) {
    const PartMesh *const mesh = solver->mesh;
    double *buffer;
    int cnt;

    /* NetCDF file id. */
    int ncid;
    /* Dimension ids. */
    int x_dimid, y_dimid, z_dimid, time_dimid, xf_dimid, yf_dimid, zf_dimid;
    /* Variable ids. */
    int x_varid, y_varid, z_varid, time_varid, iter_varid;
    int xf_varid, yf_varid, zf_varid;
    int u_varid, v_varid, w_varid, p_varid;
    /* Array of dimension ids. */
    int dimids[4];

    /* Variable values. */
    double *x_value, *y_value, *z_value;
    double time_value[1] = {0};
    int iter_value[1] = {0};
    double (*u_value)[mesh->Ny][mesh->Nx];
    double (*v_value)[mesh->Ny][mesh->Nx];
    double (*w_value)[mesh->Ny][mesh->Nx];
    double (*p_value)[mesh->Ny][mesh->Nx];

    int stat;

    buffer = calloc(4 * divceil(mesh->Nx, mesh->Px)
                      * divceil(mesh->Ny, mesh->Py)
                      * divceil(mesh->Nz, mesh->Pz),
                    sizeof(double));

    if (mesh->rank == 0) {
        x_value = calloc(mesh->Nx, sizeof(double));
        y_value = calloc(mesh->Ny, sizeof(double));
        z_value = calloc(mesh->Nz, sizeof(double));

        for (int i = 0; i < mesh->Nx; i++) {
            x_value[i] = (mesh->xf[i] + mesh->xf[i+1]) / 2;
        }
        for (int j = 0; j < mesh->Nx; j++) {
            y_value[j] = (mesh->yf[j] + mesh->yf[j+1]) / 2;
        }
        for (int k = 0; k < mesh->Nz; k++) {
            z_value[k] = (mesh->zf[k] + mesh->zf[k+1]) / 2;
        }

        u_value = calloc(mesh->Nz, sizeof(double [mesh->Ny][mesh->Nx]));
        v_value = calloc(mesh->Nz, sizeof(double [mesh->Ny][mesh->Nx]));
        w_value = calloc(mesh->Nz, sizeof(double [mesh->Ny][mesh->Nx]));
        p_value = calloc(mesh->Nz, sizeof(double [mesh->Ny][mesh->Nx]));

        for (int i = mesh->ilower; i <= mesh->iupper; i++) {
            for (int j = mesh->jlower; j <= mesh->jupper; j++) {
                for (int k = mesh->klower; k <= mesh->kupper; k++) {
                    u_value[k][j][i] = e(solver->vars->u1, i, j, k);
                    v_value[k][j][i] = e(solver->vars->u2, i, j, k);
                    w_value[k][j][i] = e(solver->vars->u3, i, j, k);
                    p_value[k][j][i] = e(solver->vars->p, i, j, k);
                }
            }
        }

        for (int r = 1; r < mesh->nprocs; r++) {
            const int ri = r / (mesh->Py * mesh->Pz);
            const int rj = r % (mesh->Py * mesh->Pz) / mesh->Pz;
            const int rk = r % mesh->Pz;

            const int ilower = mesh->Nx*ri / mesh->Px;
            const int jlower = mesh->Ny*rj / mesh->Py;
            const int klower = mesh->Nz*rk / mesh->Pz;

            const int iupper = mesh->Nx*(ri+1) / mesh->Px - 1;
            const int jupper = mesh->Ny*(rj+1) / mesh->Py - 1;
            const int kupper = mesh->Nz*(rk+1) / mesh->Pz - 1;

            MPI_Recv(buffer, 4*(iupper-ilower+1)*(jupper-jlower+1)*(kupper-klower+1), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            for (int i = ilower; i <= iupper; i++) {
                for (int j = jlower; j <= jupper; j++) {
                    for (int k = klower; k <= kupper; k++) {
                        u_value[k][j][i] = buffer[cnt++];
                        v_value[k][j][i] = buffer[cnt++];
                        w_value[k][j][i] = buffer[cnt++];
                        p_value[k][j][i] = buffer[cnt++];
                    }
                }
            }
        }

        /* Create file. */
        stat = nc_create(writer->filename, NC_CLOBBER, &ncid);
        if (stat != NC_NOERR) {
            printf("error: cannot open file %s\n", writer->filename);
            goto error;
        }

        /* Define dimensions. */
        nc_def_dim(ncid, "time", 1, &time_dimid);
        nc_def_dim(ncid, "x", mesh->Nx, &x_dimid);
        nc_def_dim(ncid, "y", mesh->Ny, &y_dimid);
        nc_def_dim(ncid, "z", mesh->Nz, &z_dimid);
        nc_def_dim(ncid, "xf", mesh->Nx+1, &xf_dimid);
        nc_def_dim(ncid, "yf", mesh->Ny+1, &yf_dimid);
        nc_def_dim(ncid, "zf", mesh->Nz+1, &zf_dimid);

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

        nc_def_var(ncid, "xf", NC_DOUBLE, 1, &xf_dimid, &xf_varid);
        nc_put_att_text(ncid, x_varid, "axis", 1, "X");
        nc_put_att_text(ncid, x_varid, "units", 6, "meters");

        nc_def_var(ncid, "yf", NC_DOUBLE, 1, &yf_dimid, &yf_varid);
        nc_put_att_text(ncid, y_varid, "axis", 1, "Y");
        nc_put_att_text(ncid, y_varid, "units", 6, "meters");

        nc_def_var(ncid, "zf", NC_DOUBLE, 1, &zf_dimid, &zf_varid);
        nc_put_att_text(ncid, z_varid, "axis", 1, "Z");
        nc_put_att_text(ncid, z_varid, "units", 6, "meters");

        nc_def_var(ncid, "iter", NC_INT, 0, NULL, &iter_varid);

        dimids[0] = time_dimid;
        dimids[1] = z_dimid;
        dimids[2] = y_dimid;
        dimids[3] = x_dimid;

        nc_def_var(ncid, "u", NC_DOUBLE, 4, dimids, &u_varid);
        nc_put_att_text(ncid, u_varid, "units", 3, "m/s");

        nc_def_var(ncid, "v", NC_DOUBLE, 4, dimids, &v_varid);
        nc_put_att_text(ncid, v_varid, "units", 3, "m/s");

        nc_def_var(ncid, "w", NC_DOUBLE, 4, dimids, &w_varid);
        nc_put_att_text(ncid, w_varid, "units", 3, "m/s");

        nc_def_var(ncid, "p", NC_DOUBLE, 4, dimids, &p_varid);
        nc_put_att_text(ncid, p_varid, "units", 3, "m/s");

        /* End of definitions. */
        nc_enddef(ncid);

        /* Write values. */
        nc_put_var_double(ncid, time_varid, time_value);
        nc_put_var_double(ncid, x_varid, x_value);
        nc_put_var_double(ncid, y_varid, y_value);
        nc_put_var_double(ncid, z_varid, z_value);
        nc_put_var_double(ncid, xf_varid, mesh->xf);
        nc_put_var_double(ncid, yf_varid, mesh->yf);
        nc_put_var_double(ncid, zf_varid, mesh->zf);
        nc_put_var_int(ncid, iter_varid, iter_value);

        nc_put_var_double(ncid, u_varid, &u_value[0][0][0]);
        nc_put_var_double(ncid, v_varid, &v_value[0][0][0]);
        nc_put_var_double(ncid, w_varid, &w_value[0][0][0]);
        nc_put_var_double(ncid, p_varid, &p_value[0][0][0]);

        /* Close file. */
        nc_close(ncid);

error:
        free(u_value);
        free(v_value);
        free(w_value);
        free(p_value);
    } else {
        /* Send to process 0. */
        cnt = 0;
        for (int i = 0; i < mesh->Nx_l; i++) {
            for (int j = 0; j < mesh->Ny_l; j++) {
                for (int k = 0; k < mesh->Nz_l; k++) {
                    buffer[cnt++] = e(solver->vars->u1, i, j, k);
                    buffer[cnt++] = e(solver->vars->u2, i, j, k);
                    buffer[cnt++] = e(solver->vars->u3, i, j, k);
                    buffer[cnt++] = e(solver->vars->p, i, j, k);
                }
            }
        }
        MPI_Send(buffer, 4*mesh->Nx_l*mesh->Ny_l*mesh->Nz_l, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void Solver_Destroy(Solver *solver) {
    // TODO: destroy solver.
}
