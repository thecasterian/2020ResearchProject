#include <stdio.h>
#include <stdbool.h>

#include "geo3d.h"
#include "ibm3d.h"

static FILE *fopen_check(
    const char *restrict filename, const char *restrict modes
);

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
    /* Names of velocity input files for initialization. */
    char init_file_u1[100], init_file_u2[100], init_file_u3[100];
    /* Names of pressure input file for initialization. */
    char init_file_p[100];

    /* Name of velocity output files for result export */
    char output_file_u1[100], output_file_u2[100], output_file_u3[100];
    /* Name of pressure output file for result export */
    char output_file_p[100];

    /* Import and export files. */
    FILE *fp_u1, *fp_u2, *fp_u3, *fp_p;

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

    /* Read initialization file names */
    fscanf(fp_in, "%*s %d", &init_using_file);
    fscanf(fp_in, "%*s %s", init_file_u1);
    fscanf(fp_in, "%*s %s", init_file_u2);
    fscanf(fp_in, "%*s %s", init_file_u3);
    fscanf(fp_in, "%*s %s", init_file_p);

    /* Read output file names */
    fscanf(fp_in, "%*s %s", output_file_u1);
    fscanf(fp_in, "%*s %s", output_file_u2);
    fscanf(fp_in, "%*s %s", output_file_u3);
    fscanf(fp_in, "%*s %s", output_file_p);

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

    IBMSolver *solver = IBMSolver_new(num_process, rank);
    IBMSolver_set_grid_params(solver, Nx, Ny, Nz, xf, yf, zf, Re, dt);
    IBMSolver_set_obstacle(solver, poly);
    Polyhedron_destroy(poly);

    /*===== Initialize flow. =================================================*/

    if (!init_using_file) {
        IBMSolver_init_flow_const(solver);
    }
    else {
        fp_u1 = fopen_check(init_file_u1, "rb");
        fp_u2 = fopen_check(init_file_u2, "rb");
        fp_u3 = fopen_check(init_file_u3, "rb");
        fp_p = fopen_check(init_file_p, "rb");

        IBMSolver_init_flow_file(solver, fp_u1, fp_u2, fp_u3, fp_p);

        fclose(fp_u1);
        fclose(fp_u2);
        fclose(fp_u3);
        fclose(fp_p);
    }

    if (rank == 0) {
        printf("\nInitialization done\n");
    }

    /*===== Run. =============================================================*/

    IBMSolver_iterate(solver, num_time_steps, true);

    /*===== Export results. ==================================================*/

    fp_u1 = fopen_check(output_file_u1, "wb");
    fp_u2 = fopen_check(output_file_u2, "wb");
    fp_u3 = fopen_check(output_file_u3, "wb");
    fp_p = fopen_check(output_file_p, "wb");

    IBMSolver_export_results(solver, fp_u1, fp_u2, fp_u3, fp_p);

    fclose(fp_u1);
    fclose(fp_u2);
    fclose(fp_u3);
    fclose(fp_p);

    /*===== Free memory and finalize. ========================================*/

    free(xf); free(yf); free(zf);
    IBMSolver_destroy(solver);

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

static FILE *fopen_check(const char *restrict filename, const char *restrict modes) {
    FILE *fp = fopen(filename, modes);
    if (!fp) {
        fprintf(stderr, "error: cannot open file \"%s\"\n", filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return fp;
}
