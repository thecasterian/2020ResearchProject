#include <stdio.h>
#include <math.h>

#include "ibm3d.h"

const int Nx = 192;
const int Ny = 160;
const int Nz = 129;

const double Re = 3300;
const double dt = 0.005;

const double PI = 3.1415926535897932;

double initfunc_u1(double, double, double);
double initfunc_u2(double, double, double);
double initfunc_u3(double, double, double);
double initfunc_p(double, double, double);

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
        zf[k] = -cos(PI * k / Nz);
    }

    solver = IBMSolver_new(num_process, rank);

    IBMSolver_set_grid(solver, Nx, Ny, Nz, xf, yf, zf);
    IBMSolver_set_params(solver, Re, dt);

    IBMSolver_set_bc(solver, DIR_WEST, BC_VELOCITY_PERIODIC, 8*PI/Re);
    IBMSolver_set_bc(solver, DIR_EAST, BC_VELOCITY_PERIODIC, 0);
    IBMSolver_set_bc(solver, DIR_SOUTH | DIR_NORTH, BC_ALL_PERIODIC, 0);
    IBMSolver_set_bc(solver, DIR_DOWN | DIR_UP, BC_STATIONARY_WALL, 0);

    IBMSolver_set_obstacle(solver, NULL);

    IBMSolver_set_linear_solver(solver, SOLVER_BiCGSTAB, PRECOND_AMG, 1e-6);

    IBMSolver_set_autosave(
        solver,
        "data/channel_u1", "data/channel_u2", "data/channel_u3", "data/channel_p",
        200
    );

    IBMSolver_assemble(solver);

    printf("%d: %d, %d\n", rank, solver->ilower, solver->iupper);

    /* Initialize. */
    if (1) {
        IBMSolver_init_flow_func(
            solver,
            initfunc_u1, initfunc_u2, initfunc_u3, initfunc_p
        );
    }
    else {
        IBMSolver_init_flow_file(
            solver,
            "data/channel_u1.out", "data/channel_u2.out", "data/channel_u3.out", "data/channel_p.out"
        );
    }

    /* Iterate. */
    IBMSolver_iterate(solver, 10, true);

    /* Export result. */
    IBMSolver_export_results(
        solver,
        "data/channel_u1", "data/channel_u2", "data/channel_u3", "data/channel_p"
    );

    /* Finalize. */
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

double initfunc_u1(double x, double y, double z) {
    return 1 - z*z;
}

double initfunc_u2(double x, double y, double z) {
    return 0;
}

double initfunc_u3(double x, double y, double z) {
    return 0;
}

double initfunc_p(double x, double y, double z) {
    return 4*PI/Re - 2/Re*x;
}
