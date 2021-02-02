#include <stdio.h>

#include <math.h>
#include "utils.h"
#include "geo3d.h"
#include "ibm3d.h"

#define PATH "/home/jeonukim/data/building"

int main(int argc, char **argv) {
    /*===== Initialize program and parse arguments. ==========================*/
    /*----- Initialize MPI. --------------------------------------------------*/

    /* Number of all processes. */
    int num_process;
    /* Rank of current process. */
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*----- Initialize HYPRE. ------------------------------------------------*/

    HYPRE_Init();

    /*----- Print run infos. -------------------------------------------------*/

    if (rank == 0) {
        printf("Run with %d processes\n", num_process);
    }

    /*===== Define variables. ================================================*/

    /* Input files. */
    FILE *fp_in, *fp_poly;

    /* Name of stl file containing polyhedron info. */
    char stl_file[100];
    /* Polyhedron read from stl file. */
    Polyhedron *poly;

    /* Number of cells in x, y, and z direction, respectively. */
    int Nx, Ny, Nz;
    /* Coordinates of cell faces. */
    double *xf, *yf, *zf;
    /* Reynolds number. */
    double Re;
    /* Delta t. */
    double dt;
    /* Total number of time steps. */
    int num_time_steps;

    /* Initialize velocities and pressure from file? (T/F) */
    int init_using_file;
    /* Input file Name for initialization. */
    char init_file[50];
    /* Output file Name for result export */
    char output_file[50];

    /*===== Read input file. =================================================*/

    /* Open input file */
    if (rank == 0) {
        printf("\nRead input file\n");
    }
    fp_in = fopen_check("ibm3d.in", "r");

    /* Read stl file */
    fscanf(fp_in, "%*s %s", stl_file);
    if (rank == 0) {
        printf("Read polyhedron file: %s\n", stl_file);
    }
    fp_poly = fopen_check(stl_file, "rb");

    poly = Polyhedron_new();
    Polyhedron_read_stl(poly, fp_poly);

    /* Read grid geometry */
    fscanf(fp_in, "%*s %d", &Nx);
    xf = calloc(Nx+1, sizeof(double));
    for (int i = 0; i <= Nx; i++) {
        fscanf(fp_in, "%lf", &xf[i]);
    }

    fscanf(fp_in, "%*s %d", &Ny);
    yf = calloc(Ny+1, sizeof(double));
    for (int j = 0; j <= Ny; j++) {
        fscanf(fp_in, "%lf", &yf[j]);
    }

    fscanf(fp_in, "%*s %d", &Nz);
    zf = calloc(Nz+1, sizeof(double));
    for (int k = 0; k <= Nz; k++) {
        fscanf(fp_in, "%lf", &zf[k]);
    }

    /* Read Reynolds number, delta t, and number of time steps */
    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &num_time_steps);

    /* Initialize with file? */
    fscanf(fp_in, "%*s %d", &init_using_file);
    /* Read initialization file name */
    fscanf(fp_in, "%*s %s", init_file);
    /* Read output file name */
    fscanf(fp_in, "%*s %s", output_file);

    fclose(fp_in);
    fclose(fp_poly);

    /*===== Print input statistics ===========================================*/

    if (rank == 0) {
        /* Mesh statistics */
        printf("\n");
        printf("Input mesh size: %d x %d x %d\n", Nx, Ny, Nz);
        printf("  xmin: %10.4lf, xmax: %10.4lf\n", xf[0], xf[Nx]);
        printf("  ymin: %10.4lf, ymax: %10.4lf\n", yf[0], yf[Ny]);
        printf("  zmin: %10.4lf, zmax: %10.4lf\n", zf[0], zf[Ny]);

        /* Polyhedron statistics */
        Polyhedron_print_stats(poly);

        /* Reynolds number and delta t */
        printf("\n");
        printf("Reynolds no.: %.6lf\n", Re);
        printf("delta t     : %.6lf\n", dt);
    }

    /*===== Set solver. ======================================================*/

    IBMSolver *solver = IBMSolver_new(num_process, rank, 16, 1, 1);

    IBMSolver_set_grid(solver, Nx, Ny, Nz, xf, yf, zf);
    IBMSolver_set_params(solver, Re, dt);

    IBMSolver_set_bc(solver, DIR_WEST, BC_VELOCITY_COMPONENT, VAL_CONST, 1., 0., 0.);
    IBMSolver_set_bc(solver, DIR_EAST, BC_PRESSURE, VAL_CONST, 0.);
    IBMSolver_set_bc(solver, DIR_NORTH | DIR_SOUTH | DIR_DOWN | DIR_UP, BC_FREE_SLIP_WALL);

    IBMSolver_set_obstacle(solver, poly);

    IBMSolver_set_linear_solver(solver, SOLVER_BiCGSTAB, PRECOND_AMG, 1e-6);

    IBMSolver_set_autosave(solver, output_file, 0);

    IBMSolver_assemble(solver);

    Polyhedron_destroy(poly);

    /*===== Initialize flow. =================================================*/

    if (!init_using_file) {
        IBMSolver_init_flow_const(solver, 1, 0, 0, 0);
    }
    else {
        IBMSolver_init_flow_file(solver, init_file);
    }

    IBMSolver_export_lvset_flag(solver, PATH "/lvset_flag");

    if (rank == 0) {
        printf("\nInitialization done\n");
    }

    /*===== Run. =============================================================*/

    IBMSolver_iterate(solver, num_time_steps, true);

    /*===== Export results. ==================================================*/

    IBMSolver_export_result(solver, output_file);

    /*===== Free memory and finalize. ========================================*/

    free(xf); free(yf); free(zf);
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}
