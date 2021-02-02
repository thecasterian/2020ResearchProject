#include <stdio.h>
#include <math.h>

#include "geo3d.h"
#include "ibm3d.h"
#include "utils.h"

#define Nx 1000
#define Ny 160
#define Nz 150

#define PATH "/home/jeonukim/data/building"

const double Re = 500;
const double dt = 0.005;

/* Grid coordinates. */
double xf[Nx+1], yf[Ny+1], zf[Nz+1];

double initfunc_u1(double, double, double);
double initfunc_u2(double, double, double);
double initfunc_u3(double, double, double);
double initfunc_p(double, double, double);

double inlet_vel_x(double, double, double, double);
double inlet_vel_y(double, double, double, double);
double inlet_vel_z(double, double, double, double);

int main(int argc, char **argv) {
    /* Number of all processes. */
    int num_process;
    /* Rank of current process. */
    int rank;
    /* Building STL. */
    FILE *fp_building = fopen_check("/home/jeonukim/data/lotte.stl", "rb");
    Polyhedron *poly = Polyhedron_new();
    /* IBM solver. */
    IBMSolver *solver;

    for (int i = 0; i <= Nx; i++) {
        xf[i] = -600 + (1400. / Nx)*i;
    }
    for (int j = 0; j <= Ny; j++) {
        yf[j] = -550 + (1000. / Ny)*j;
    }
    for (int k = 0; k <= Nz; k++) {
        zf[k] = (300. / Nz)*k;
    }

    /* Initialize. */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    HYPRE_Init();

    if (rank == 0) {
        printf("Run with %d processes\n", num_process);
    }

    /* Read building STL file. */
    Polyhedron_read_stl(poly, fp_building);

    /* Set solver. */
    solver = IBMSolver_new(num_process, rank, 16, 1, 1);

    IBMSolver_set_grid(solver, Nx, Ny, Nz, xf, yf, zf);
    IBMSolver_set_params(solver, Re, dt);

    IBMSolver_set_bc(solver, DIR_WEST, BC_VELOCITY_COMPONENT, VAL_FUNC, inlet_vel_x, inlet_vel_y, inlet_vel_z);
    IBMSolver_set_bc(solver, DIR_EAST, BC_PRESSURE, VAL_CONST, 0.);
    IBMSolver_set_bc(solver, DIR_SOUTH | DIR_NORTH | DIR_UP, BC_FREE_SLIP_WALL);
    IBMSolver_set_bc(solver, DIR_DOWN, BC_STATIONARY_WALL);

    // IBMSolver_set_turb_model(solver, TURBMODEL_SMAGORINSKY, 0.17);

    IBMSolver_set_obstacle(solver, poly);

    IBMSolver_set_linear_solver(solver, SOLVER_BiCGSTAB, PRECOND_AMG, 1e-5);

    IBMSolver_set_autosave(solver, PATH "/lotte", 10);

    IBMSolver_assemble(solver);

    IBMSolver_export_lvset_flag(solver, PATH "/lvset_flag");

    /* Print statistics and problem info. */
    if (rank == 0) {
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
    if (1) {
        IBMSolver_init_flow_func(
            solver,
            initfunc_u1, initfunc_u2, initfunc_u3, initfunc_p
        );
    }
    else {
        IBMSolver_init_flow_file(solver, PATH "/lotte");
    }

    /* Iterate. */
    IBMSolver_iterate(solver, 10, true);

    IBMSolver_export_intermediate(solver, PATH "/inter");

    /* Export result. */
    IBMSolver_export_result(solver, PATH "/lotte");

    /* Finalize. */
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

double initfunc_u1(double x UNUSED, double y UNUSED, double z) {
    return pow(z/300, 1./7);
}

double initfunc_u2(double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

double initfunc_u3(double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

double initfunc_p(double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

double inlet_vel_x(double t UNUSED, double x UNUSED, double y UNUSED, double z) {
    return pow(z/300, 1./7);
}

double inlet_vel_y(double t UNUSED, double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}

double inlet_vel_z(double t UNUSED, double x UNUSED, double y UNUSED, double z UNUSED) {
    return 0;
}
