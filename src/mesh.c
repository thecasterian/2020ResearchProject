#include "../include/mesh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../include/utils.h"
#include "../include/netcdf_io.h"

static void calc_part(PartMesh *mesh);
static void arr_alloc(PartMesh *mesh);

PartMesh *PartMesh_Create(int nprocs, int rank,
                          int Px, int Py, int Pz) {
    PartMesh *mesh;

    if (Px * Py * Pz != nprocs) {
        printf("error: #processes does not match (%d * %d * %d != %d)\n",
               Px, Py, Pz, nprocs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (rank < 0 || rank >= nprocs) {
        printf("error: invalid rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    mesh = calloc(1, sizeof(*mesh));

    mesh->nprocs = nprocs;
    mesh->rank = rank;

    mesh->Px = Px;
    mesh->Py = Py;
    mesh->Pz = Pz;
    mesh->ri = rank / (Py * Pz);
    mesh->rj = rank % (Py * Pz) / Pz;
    mesh->rk = rank % Pz;
    mesh->prevx = mesh->ri != 0 ? rank - Py*Pz : rank + (Px-1)*Py*Pz;
    mesh->prevy = mesh->rj != 0 ? rank - Pz : rank + (Py-1)*Pz;
    mesh->prevz = mesh->rk != 0 ? rank - 1 : rank + (Pz-1);
    mesh->nextx = mesh->ri != Px-1 ? rank + Py*Pz : rank - (Px-1)*Py*Pz;
    mesh->nexty = mesh->rj != Py-1 ? rank + Pz : rank - (Py-1)*Pz;
    mesh->nextz = mesh->rk != Pz-1 ? rank + 1 : rank - (Pz-1);

    return mesh;
}

void PartMesh_ReadCoord(PartMesh *mesh,
                        int Nx, int Ny, int Nz,
                        const double *xf, const double *yf, const double *zf) {
    mesh->Nx = Nx;
    mesh->Ny = Ny;
    mesh->Nz = Nz;

    calc_part(mesh);
    arr_alloc(mesh);

    memcpy(mesh->xf+3, xf, (Nx+1)*sizeof(double));
    memcpy(mesh->yf+3, yf, (Ny+1)*sizeof(double));
    memcpy(mesh->zf+3, zf, (Nz+1)*sizeof(double));
}

void PartMesh_ReadNetCDF(PartMesh *mesh, NetCDFReader *reader) {
    mesh->Nx = reader->Nx;
    mesh->Ny = reader->Ny;
    mesh->Nz = reader->Nz;

    calc_part(mesh);
    arr_alloc(mesh);

    memcpy(mesh->xf, reader->xf, (reader->Nx+7)*sizeof(double));
    memcpy(mesh->yf, reader->yf, (reader->Ny+7)*sizeof(double));
    memcpy(mesh->zf, reader->zf, (reader->Nz+7)*sizeof(double));
}

void PartMesh_CalcTransform(PartMesh *mesh) {
    double a, b;

    /* Calculate the coordinates of the outer cell faces. */
    e(mesh->xf, -3) = 2*e(mesh->xf, 0) - e(mesh->xf, 3);
    e(mesh->xf, -2) = 2*e(mesh->xf, 0) - e(mesh->xf, 2);
    e(mesh->xf, -1) = 2*e(mesh->xf, 0) - e(mesh->xf, 1);
    e(mesh->xf, mesh->Nx+1) = 2*e(mesh->xf, mesh->Nx) - e(mesh->xf, mesh->Nx-1);
    e(mesh->xf, mesh->Nx+2) = 2*e(mesh->xf, mesh->Nx) - e(mesh->xf, mesh->Nx-2);
    e(mesh->xf, mesh->Nx+3) = 2*e(mesh->xf, mesh->Nx) - e(mesh->xf, mesh->Nx-3);
    e(mesh->yf, -3) = 2*e(mesh->yf, 0) - e(mesh->yf, 3);
    e(mesh->yf, -2) = 2*e(mesh->yf, 0) - e(mesh->yf, 2);
    e(mesh->yf, -1) = 2*e(mesh->yf, 0) - e(mesh->yf, 1);
    e(mesh->yf, mesh->Ny+1) = 2*e(mesh->yf, mesh->Ny) - e(mesh->yf, mesh->Ny-1);
    e(mesh->yf, mesh->Ny+2) = 2*e(mesh->yf, mesh->Ny) - e(mesh->yf, mesh->Ny-2);
    e(mesh->yf, mesh->Ny+3) = 2*e(mesh->yf, mesh->Ny) - e(mesh->yf, mesh->Ny-3);
    e(mesh->zf, -3) = 2*e(mesh->zf, 0) - e(mesh->zf, 3);
    e(mesh->zf, -2) = 2*e(mesh->zf, 0) - e(mesh->zf, 2);
    e(mesh->zf, -1) = 2*e(mesh->zf, 0) - e(mesh->zf, 1);
    e(mesh->zf, mesh->Nz+1) = 2*e(mesh->zf, mesh->Nz) - e(mesh->zf, mesh->Nz-1);
    e(mesh->zf, mesh->Nz+2) = 2*e(mesh->zf, mesh->Nz) - e(mesh->zf, mesh->Nz-2);
    e(mesh->zf, mesh->Nz+3) = 2*e(mesh->zf, mesh->Nz) - e(mesh->zf, mesh->Nz-3);

    /* Calculate the grid transformation. */
    for (int i = -2; i <= mesh->Nx_l+2; i++) {
        a = e(mesh->xf, mesh->ilower+i) - e(mesh->xf, mesh->ilower+i-1);
        b = e(mesh->xf, mesh->ilower+i+1) - e(mesh->xf, mesh->ilower+i);
        e(mesh->dxi_dx_f, i) = (a*a+b*b) / (a*b*(a+b));
    }
    for (int j = -1; j <= mesh->Ny_l+1; j++) {
        a = e(mesh->yf, mesh->jlower+j) - e(mesh->yf, mesh->jlower+j-1);
        b = e(mesh->yf, mesh->jlower+j+1) - e(mesh->yf, mesh->jlower+j);
        e(mesh->deta_dy_f, j) = (a*a+b*b) / (a*b*(a+b));
    }
    for (int k = -1; k <= mesh->Nz_l+1; k++) {
        a = e(mesh->zf, mesh->klower+k) - e(mesh->zf, mesh->klower+k-1);
        b = e(mesh->zf, mesh->klower+k+1) - e(mesh->zf, mesh->klower+k);
        e(mesh->dzeta_dz_f, k) = (a*a+b*b) / (a*b*(a+b));
    }
    for (int i = -3; i <= mesh->Nx_l+2; i++)
        e(mesh->dxi_dx_c, i) = 1 / (e(mesh->xf, mesh->ilower+i+1)
                                    - e(mesh->xf, mesh->ilower+i));
    for (int j = -3; j <= mesh->Ny_l+2; j++)
        e(mesh->deta_dy_c, j) = 1 / (e(mesh->yf, mesh->jlower+j+1)
                                     - e(mesh->yf, mesh->jlower+j));
    for (int k = -3; k <= mesh->Nz_l+2; k++)
        e(mesh->dzeta_dz_c, k) = 1 / (e(mesh->zf, mesh->klower+k+1)
                                      - e(mesh->zf, mesh->klower+k));

    /* Calculate the Jacobian. */
    FOR_U_ALL (i, j, k)
        e(mesh->J, i, j, k) = e(mesh->dxi_dx_c, i) * e(mesh->deta_dy_c, j)
                              * e(mesh->dzeta_dz_c, k);
}

void PartMesh_Destroy(PartMesh *mesh) {
    free(mesh->xf);
    free(mesh->yf);
    free(mesh->zf);
    free(mesh->dxi_dx_f);
    free(mesh->deta_dy_f);
    free(mesh->dzeta_dz_f);
    free(mesh->dxi_dx_c);
    free(mesh->deta_dy_c);
    free(mesh->dzeta_dz_c);
    free(mesh->J);
    free(mesh);
}

static void calc_part(PartMesh *mesh) {
    mesh->ilower = mesh->Nx * mesh->ri / mesh->Px;
    mesh->jlower = mesh->Ny * mesh->rj / mesh->Py;
    mesh->klower = mesh->Nz * mesh->rk / mesh->Pz;
    mesh->iupper = mesh->Nx * (mesh->ri+1) / mesh->Px - 1;
    mesh->jupper = mesh->Ny * (mesh->rj+1) / mesh->Py - 1;
    mesh->kupper = mesh->Nz * (mesh->rk+1) / mesh->Pz - 1;
    mesh->Nx_l = mesh->iupper - mesh->ilower + 1;
    mesh->Ny_l = mesh->jupper - mesh->jlower + 1;
    mesh->Nz_l = mesh->kupper - mesh->klower + 1;
}

static void arr_alloc(PartMesh *mesh) {
    mesh->xf = calloc(mesh->Nx+7, sizeof(double));
    mesh->yf = calloc(mesh->Ny+7, sizeof(double));
    mesh->zf = calloc(mesh->Nz+7, sizeof(double));
    mesh->dxi_dx_f = calloc(mesh->Nx+7, sizeof(double));
    mesh->deta_dy_f = calloc(mesh->Ny+7, sizeof(double));
    mesh->dzeta_dz_f = calloc(mesh->Nz+7, sizeof(double));
    mesh->dxi_dx_c = calloc(mesh->Nx+6, sizeof(double));
    mesh->deta_dy_c = calloc(mesh->Ny+6, sizeof(double));
    mesh->dzeta_dz_c = calloc(mesh->Nz+6, sizeof(double));
    mesh->J = calloc((mesh->Nx+6)*(mesh->Ny+6)*(mesh->Nz+6), sizeof(double));
}
