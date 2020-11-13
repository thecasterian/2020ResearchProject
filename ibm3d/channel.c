#include <stdio.h>
#include <math.h>

#include "utils.h"
#include "geo3d.h"
#include "ibm3d.h"

const int Nx = 192;
const int Ny = 160;
const int Nz = 129;

const double Re = 3300;
const double dt = 0.005;

const double PI = 3.1415926535897932;

int main(int argc, char **argv) {
    /* Number of all processes. */
    int num_process;
    /* Rank of current process. */
    int rank;
    /* Grid coordinates. */
    double xf[Nx+1], yf[Ny+1], zf[Nz+1];
    /* IBM solver. */
    IBMSolver *solver;

    /* Initialize. */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    HYPRE_Init();

    if (rank == 0) {
        printf("Run with %d processes\n", num_process);
    }

    /* Set solver. */
    for (int i = 0; i <= Nx; i++) {
        xf[i] = 4*PI/Nx * i - 2*PI;
    }
    for (int j = 0; j <= Ny; j++) {
        yf[j] = 2*PI/Ny * j - PI;
    }
    for (int k = 0; k <= Nz; k++) {
        zf[k] = -2*cos(PI * k / Nz);
    }

    solver = IBMSolver_new(num_process, rank);

    IBMSolver_set_grid(solver, Nx, Ny, Nz, xf, yf, zf);
    IBMSolver_set_params(solver, Re, dt);

    IBMSolver_set_bc(solver, DIR_WEST, BC_VELOCITY_INLET, 1);
    IBMSolver_set_bc(solver, DIR_EAST, BC_PRESSURE_OUTLET, 1);
    IBMSolver_set_bc(solver, DIR_SOUTH | DIR_NORTH, BC_ALL_PERIODIC, 0);
    IBMSolver_set_bc(solver, DIR_DOWN | DIR_UP, BC_STATIONARY_WALL, 0);

    IBMSolver_set_obstacle(solver, NULL);

    IBMSolver_set_linear_solver(solver, SOLVER_BiCGSTAB, PRECOND_AMG, 1e-6);

    IBMSolver_assemble(solver);

    /* Initialize. */
    IBMSolver_init_flow_const(solver);
    // IBMSolver_init_flow_file(
    //     solver,
    //     "channel_u1.out", "channel_u2.out", "channel_u3.out", "channel_p.out"
    // );

    /* Iterate. */
    IBMSolver_iterate(solver, 10, true);

    /* Export result. */
    IBMSolver_export_results(
        solver,
        "channel_u1.out", "channel_u2.out", "channel_u3.out", "channel_p.out"
    );

    /* Finalize. */
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}
