#include <stdio.h>
#include <math.h>

#include "include/ibm3d.h"
#include <mpi.h>

const int Nx = 192;
const int Ny = 160;
const int Nz = 129;

const double delta = 1;

const double PI = 3.141592653589793238;

double init_u1(double x UNUSED, double y UNUSED, double z) {
    return 1-(z/delta)*(z/delta);
}

double init_u2(double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

double init_u3(double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

double init_p(double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

int main(int argc, char *argv[]) {
    int nprocs, rank;
    double xf[Nx+1], yf[Ny+1], zf[Nz+1];

    PartMesh *mesh;
    Params params = {1, 1, 0.01};
    Solver *solver;
    Initializer *init;
    NetCDFWriter *writer;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i <= Nx; i++) {
        xf[i] = 4*PI*delta * (double)i/Nx - 2*PI*delta;
    }
    for (int j = 0; j <= Ny; j++) {
        yf[j] = 2*PI*delta * (double)j/Ny - PI*delta;
    }
    for (int k = 0; k <= Nz; k++) {
        zf[k] = -delta * cos(PI * (double)k/Nz);
    }

    mesh = PartMesh_Create(nprocs, rank, 4, 2, 2);
    PartMesh_ReadCoord(mesh, Nx, Ny, Nz, xf, yf, zf);

    solver = Solver_Create(mesh, &params);
    Solver_Assemble(solver);

    init = Initializer_Create(INIT_FUNC, init_u1, init_u2, init_u3, init_p);
    Solver_Initialize(solver, init);

    Solver_Iterate(solver, 10, true);

    writer = NetCDFWriter_Create("/home/jeonukim/data/test");
    Solver_ExportNetCDF(solver, writer);

    PartMesh_Destroy(mesh);
    Solver_Destroy(solver);
    Initializer_Destroy(init);

    MPI_Finalize();
}
