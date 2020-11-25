#include <stdio.h>
#include <math.h>

#include "geo3d.h"
#include "ibm3d.h"

const int Nx = 288;
const int Ny = 144;
const int Nz = 144;

const double Re = 200;
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
    /* Building STL. */
    FILE *fp_building = fopen("../stl/building.stl", "rb");
    Polyhedron *poly = Polyhedron_new();
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
        xf[i] = 24. / Nx * i;
    }
    for (int j = 0; j <= Ny; j++) {
        yf[j] = 16. / Ny * j;
    }
    for (int k = 0; k <= Nz; k++) {
        zf[k] = 12. / Nz * k - 2;
    }

    /* Read building STL file. */
    Polyhedron_read_stl(poly, fp_building);
    Polyhedron_scale(poly, SCALE_INCH_TO_MM);

    /* Set solver. */
    solver = IBMSolver_new(num_process, rank);

    IBMSolver_set_grid(solver, Nx, Ny, Nz, xf, yf, zf);
    IBMSolver_set_params(solver, Re, dt);

    IBMSolver_set_bc(solver, DIR_WEST, BC_VELOCITY_INLET, 1);
    IBMSolver_set_bc(solver, DIR_EAST, BC_PRESSURE_OUTLET, 0);
    IBMSolver_set_bc(solver, DIR_SOUTH | DIR_NORTH | DIR_UP, BC_FREE_SLIP_WALL, 0);
    IBMSolver_set_bc(solver, DIR_DOWN, BC_STATIONARY_WALL, 0);

    IBMSolver_set_obstacle(solver, poly);

    IBMSolver_set_linear_solver(solver, SOLVER_BiCGSTAB, PRECOND_AMG, 1e-6);

    // IBMSolver_set_autosave(
    //     solver,
    //     "data/channel_u1", "data/channel_u2", "data/channel_u3", "data/channel_p",
    //     200
    // );

    IBMSolver_assemble(solver);

    IBMSolver_export_flag(solver, "data/flag.out");
    IBMSolver_export_lvset(solver, "data/lvset.out");

    /* Print statistics and problem info. */
    if (rank == 1) {
        /* Mesh statistics */
        printf("\n");
        printf("Input mesh size: %d x %d x %d\n", Nx, Ny, Nz);
        printf("  xmin: %10.4lf, xmax: %10.4lf\n", xf[0], xf[Nx]);
        printf("  ymin: %10.4lf, ymax: %10.4lf\n", yf[0], yf[Ny]);
        printf("  zmin: %10.4lf, zmax: %10.4lf\n", zf[0], zf[Nz]);

        /* Polyhedron statistics */
        Polyhedron_print_stats(poly);

        /* Reynolds number and delta t */
        printf("\n");
        printf("Reynolds no.: %.6lf\n", Re);
        printf("delta t     : %.6lf\n", dt);
    }

    /* Initialize. */
    if (0) {
        IBMSolver_init_flow_func(
            solver,
            initfunc_u1, initfunc_u2, initfunc_u3, initfunc_p
        );
    }
    else {
        IBMSolver_init_flow_file(
            solver,
            "data/building_u1.out", "data/building_u2.out", "data/building_u3.out", "data/building_p.out"
        );
    }

    /* Iterate. */
    IBMSolver_iterate(solver, 30, true);

    /* Export result. */
    IBMSolver_export_results(
        solver,
        "data/building_u1", "data/building_u2", "data/building_u3", "data/building_p"
    );

    /* Finalize. */
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

double initfunc_u1(double x, double y, double z) {
    return 1;
}

double initfunc_u2(double x, double y, double z) {
    return 0;
}

double initfunc_u3(double x, double y, double z) {
    return 0;
}

double initfunc_p(double x, double y, double z) {
    return 0;
}
