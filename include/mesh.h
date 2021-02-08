#ifndef IBM3D_MESH_H
#define IBM3D_MESH_H

typedef struct _netcdf_reader NetCDFReader;

typedef struct _part_mesh {
    int nprocs;                 /* Number of processes. */
    int rank;                   /* Rank of current process. */

    int Px, Py, Pz;             /* Number of processes in each direction. */
    int ri, rj, rk;             /* Rank of current process in each direction. */

    int prevx, prevy, prevz;    /* Rank of previous process in each direction. */
    int nextx, nexty, nextz;    /* Rank of next process in each direction. */

    int Nx_l, Ny_l, Nz_l;       /* Local number of cells. */
    int Nx, Ny, Nz;             /* Global number of cells. */
    int ilower, jlower, klower; /* Min indices of current process. */
    int iupper, jupper, kupper; /* Max indices of current process. */

    double *xf, *yf, *zf;       /* Local face coordinates. */

    double *dxi_dx_f, *deta_dy_f, *dzeta_dz_f;
    double *dxi_dx_c, *deta_dy_c, *dzeta_dz_c;
    double *J;
} PartMesh;

PartMesh *PartMesh_Create(int nprocs, int rank,
                          int Px, int Py, int Pz);
void PartMesh_ReadCoord(PartMesh *mesh,
                        int Nx, int Ny, int Nz,
                        const double *xf, const double *yf, const double *zf);
void PartMesh_ReadNetCDF(PartMesh *mesh, NetCDFReader *reader);

void PartMesh_CalcTransform(PartMesh *mesh);

void PartMesh_Destroy(PartMesh *mesh);

#endif
