#include <stdio.h>
#include <math.h>

#include "geo3d.h"
#include "ibm3d.h"
#include "utils.h"

#define Nx 128
#define Ny 128
#define Nz 128

#define PATH "/home/jeonukim/data/cavity"

const double Re = 3200;
const double dt = 0.005;

/* Grid coordinates. */
double xf[Nx+1], yf[Ny+1], zf[Nz+1];

int main(int argc, char **argv) {
    /* Number of all processes. */
    int num_process;
    /* Rank of current process. */
    int rank;
    /* IBM solver. */
    IBMSolver *solver;

    for (int i = 0; i <= Nx; i++) {
        xf[i] = (double)i / Nx;
    }
    for (int j = 0; j <= Ny; j++) {
        yf[j] = (double)j / Ny;
    }
    for (int k = 0; k <= Nz; k++) {
        zf[k] = (double)k / Nz;
    }

    /* Initialize. */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    HYPRE_Init();

    if (rank == 0) {
        printf("Run with %d processes\n", num_process);
    }

    /* Set solver. */
    solver = IBMSolver_new(num_process, rank, 16, 1, 1);

    IBMSolver_set_grid(solver, Nx, Ny, Nz, xf, yf, zf);
    IBMSolver_set_params(solver, Re, dt);

    IBMSolver_set_bc(solver, DIR_UP, BC_VELOCITY_COMPONENT, VAL_CONST, 1., 0., 0.);
    IBMSolver_set_bc(solver, DIR_WEST | DIR_EAST | DIR_SOUTH | DIR_NORTH | DIR_DOWN, BC_STATIONARY_WALL);

    IBMSolver_set_obstacle(solver, NULL);

    IBMSolver_set_linear_solver(solver, SOLVER_BiCGSTAB, PRECOND_AMG, 1e-6);

    IBMSolver_set_autosave(solver, PATH "/cavity", 100);

    IBMSolver_assemble(solver);

    /* Print statistics and problem info. */
    if (rank == 0) {
        /* Mesh statistics */
        printf("\n");
        printf("Input mesh size: %d x %d x %d\n", Nx, Ny, Nz);
        printf("  xmin: %10.4lf, xmax: %10.4lf\n", xf[0], xf[Nx]);
        printf("  ymin: %10.4lf, ymax: %10.4lf\n", yf[0], yf[Ny]);
        printf("  zmin: %10.4lf, zmax: %10.4lf\n", zf[0], zf[Nz]);

        /* Reynolds number and delta t */
        printf("\n");
        printf("Reynolds no.: %.6lf\n", Re);
        printf("delta t     : %.6lf\n", dt);
    }

    /* Initialize. */
    if (0) {
        IBMSolver_init_flow_const(solver, 0, 0, 0, 0);
    }
    else {
        IBMSolver_init_flow_file(solver, PATH "/cavity");
    }

    /* Iterate. */
    IBMSolver_iterate(solver, 10000, true);

    /* Export result. */
    IBMSolver_export_result(solver, PATH "/cavity");

    /* Finalize. */
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}
