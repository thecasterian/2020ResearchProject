#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include "geo3d.h"

#define ALLOC3D(arrname, nx, ny, nz) \
    double (*arrname)[ny][nz] = calloc(nx, sizeof(double [ny][nz]))
#define FOR_ALL_CELL(i, j, k) \
    for (int i = 0; i <= Nx+1; i++) \
        for (int j = 0; j <= Ny+1; j++) \
            for (int k = 0; k <= Nz+1; k++)
#define FOR_INNER_CELL(i, j, k) \
    for (int i = 1; i <= Nx; i++) \
        for (int j = 1; j <= Ny; j++) \
            for (int k = 1; k <= Nz; k++)
#define SWAP(a, b) do {double (*tmp)[] = a; a = b; b = tmp;} while (1)

/* Index of adjacent cells in 3-d cartesian coordinate */
const int adj[6][3] = {
    {0, 1, 0}, {1, 0, 0}, {0, -1, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, 1}
};
/* Order is:
            5
          z |  0
            | / y
            |/
 3 ---------+--------- 1
           /|        x
          / |
         2  |
            4
*/

/* Find the index of the first element in `arr` which is greater than `val`.
   `arr` must be sorted in increasing order. */
int upper_bound(const int len, double arr[const static len], const double val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val >= arr[mid]) {
            l = mid + 1;
        } else {
            h = mid;
        }
    }
    return l;
}

int main(int argc, char **argv) {
    /****** Initialize program and parse arguments ****************************/
    /*===== Initialize MPI ===================================================*/

    /* Id. of current process */
    int myid;
    /* Number of all processes */
    int num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /*===== Initialize HYPRE =================================================*/

    HYPRE_Init();

    /* For the present, the number of process must be 1 */
    if (num_procs != 1) {
        if (myid == 0)
            printf("Must run with 1 processors!\n");
        MPI_Finalize();
        return 0;
    }

    /****** Read input file ***************************************************/
    /*===== Define input parameters ==========================================*/

    /* Name of stl file containing polyhedron info. */
    char stl_file[100];
    /* Polyhedron read from stl file */
    Polyhedron *poly;

    /* Number of cells in x, y, and z direction, respectively */
    int Nx, Ny, Nz;
    /* Reynolds number */
    double Re;
    /* Delta t */
    double dt;
    /* Totla number of time steps */
    int numtstep;

    /* Initialize velocities and pressure from file? (T/F) */
    int init_using_file;
    /* Name of velocity input files for initialization */
    char init_file_u1[100], init_file_u2[100], init_file_u3[100];
    /* Name of pressure input file for initialization */
    char init_file_p[100];

    /* Name of velocity output files for result export */
    char output_file_u1[100], output_file_u2[100], output_file_u3[100];
    /* Name of pressure output file for result export */
    char output_file_p[100];

    /*===== Read inputs ======================================================*/

    /* Open input file */
    printf("Read input file\n");
    FILE *fp_in = fopen("ibm3d.in", "r");
    if (!fp_in) {
        fprintf(stderr, "Error: cannot open input file\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Read stl file */
    fscanf(fp_in, "%*s %s", stl_file);
    printf("Read polyhedron file: %s\n", stl_file);
    FILE *fp_poly = fopen(stl_file, "rb");
    if (!fp_poly) {
        fprintf(stderr, "Error: cannot open polygon file\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    poly = Polyhedron_new();
    Polyhedron_read_stl(poly, fp_poly);

    /* Read grid geometry */
    fscanf(fp_in, "%*s %d", &Nx);
    double xf[Nx+1];
    for (int i = 0; i <= Nx; i++) {
        fscanf(fp_in, "%lf", &xf[i]);
    }

    fscanf(fp_in, "%*s %d", &Ny);
    double yf[Ny+1];
    for (int j = 0; j <= Ny; j++) {
        fscanf(fp_in, "%lf", &yf[j]);
    }

    fscanf(fp_in, "%*s %d", &Nz);
    double zf[Nz+1];
    for (int k = 0; k <= Nz; k++) {
        fscanf(fp_in, "%lf", &zf[k]);
    }

    /* Read Reynolds number, delta t, and number of time steps */
    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep);

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

    /****** Print input statistics ********************************************/

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

    /****** Define variables **************************************************/
    /*===== Grid variables ===================================================*/

    /* Cell widths */
    double dx[Nx+2], dy[Ny+2], dz[Nz+2];
    /* Cell centroid coordinates */
    double xc[Nx+2], yc[Ny+2], zc[Nz+2];

    /* Level set function (signed distance function) at cell centroids */
    double (*const lvset)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    /* Flag of each cell (1: fluid cell, 2: ghost cell, 0: solid cell) */
    int (*const flag)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));
    /* Total number of fluid and ghost cells */
    int num_tc = 0;
    /* Number of ghost cells */
    int num_gc = 0;

    /* Cell id. (starts from 0; only assigned for fluid and ghost cells) */
    int (*cell_id)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));
    FOR_ALL_CELL (i, j, k) {
        cell_id[i][j][k] = -1;
    }
    /* Cell id. of adjacent cells; refer global variable `adj` for the order of
       adjacent cells */
    int (*adj_cell_id)[Ny+2][Nz+2][6]
        = calloc(Nx+2, sizeof(int [Ny+2][Nz+2][6]));

    /* Cell id. of 8 adjacent cells used for the trilinear interpolation of
       mirror point of the ghost cell */
    int (*gc_interp_cell_id)[8];
    /* Trilinear interpolation coefficient; order is same with
       `gc_interp_cell_id`  */
    double (*gc_interp_coeff)[8];
    /* Order of cells is:
         011         111
          +-----------+
     001 /|      101 /|          z
        +-----------+ |          | y
        | |         | |          |/
        | +---------|-+          +------ x
        |/ 010      |/ 110
        +-----------+
       000         100
    */

    /*===== Pressure and velocities ==========================================*/

    ALLOC3D(      p       , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      p_next  , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const p_prime , Nx+2, Ny+2, Nz+2);

    ALLOC3D(      u1      , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      u1_next , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u1_star , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u1_tilde, Nx+2, Ny+2, Nz+2);

    ALLOC3D(      u2      , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      u2_next , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u2_star , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u2_tilde, Nx+2, Ny+2, Nz+2);

    ALLOC3D(      u3      , Nx+2, Ny+2, Nz+2);
    ALLOC3D(      u3_next , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u3_star , Nx+2, Ny+2, Nz+2);
    ALLOC3D(const u3_tilde, Nx+2, Ny+2, Nz+2);

    ALLOC3D(      U1      , Nx+1, Ny+2, Nz+2);
    ALLOC3D(      U1_next , Nx+1, Ny+2, Nz+2);
    ALLOC3D(const U1_star , Nx+1, Ny+2, Nz+2);

    ALLOC3D(      U2      , Nx+2, Ny+1, Nz+2);
    ALLOC3D(      U2_next , Nx+2, Ny+1, Nz+2);
    ALLOC3D(const U2_star , Nx+2, Ny+1, Nz+2);

    ALLOC3D(      U3      , Nx+2, Ny+2, Nz+1);
    ALLOC3D(      U3_next , Nx+2, Ny+2, Nz+1);
    ALLOC3D(const U3_star , Nx+2, Ny+2, Nz+1);

    /*===== Fluxes ===========================================================*/

    ALLOC3D(N1     , Nx+2, Ny+2, Nz+2);
    ALLOC3D(N1_prev, Nx+2, Ny+2, Nz+2);
    ALLOC3D(N2     , Nx+2, Ny+2, Nz+2);
    ALLOC3D(N2_prev, Nx+2, Ny+2, Nz+2);
    ALLOC3D(N3     , Nx+2, Ny+2, Nz+2);
    ALLOC3D(N3_prev, Nx+2, Ny+2, Nz+2);

    /*===== Others ===========================================================*/

    double kx_W[Nx+2], kx_E[Nx+2];
    double ky_S[Ny+2], ky_N[Ny+2];
    double kz_D[Nz+2], kz_U[Nz+2];

    /*===== HYPRE matrices, vectors, solvers, and arrays =====================*/

    HYPRE_IJMatrix     A_u1, A_u2, A_u3;
    HYPRE_ParCSRMatrix parcsr_A_u1, parcsr_A_u2, parcsr_A_u3;
    HYPRE_IJVector     b_u1, b_u2, b_u3;
    HYPRE_ParVector    par_b_u1, par_b_u2, par_b_u3;
    HYPRE_IJVector     x_u1, x_u2, x_u3;
    HYPRE_ParVector    par_x_u1, par_x_u2, par_x_u3;

    HYPRE_IJMatrix     A_p;
    HYPRE_ParCSRMatrix parcsr_A_p;
    HYPRE_IJVector     b_p;
    HYPRE_ParVector    par_b_p;
    HYPRE_IJVector     x_p;
    HYPRE_ParVector    par_x_p;

    HYPRE_Solver solver_u1, solver_u2, solver_u3;
    HYPRE_Solver solver_p;
    HYPRE_Solver precond;

    int *vector_rows;
    double *vector_values_u1, *vector_values_u2, *vector_values_u3;
    double *vector_values_p;
    double *vector_zeros, *vector_res;

    int num_iters;
    double final_res_norm;

    /****** Calculate grid variables ******************************************/

    /* Cell width and centroid coordinates */
    for (int i = 1; i <= Nx; i++) {
        dx[i] = xf[i] - xf[i-1];
        xc[i] = (xf[i] + xf[i-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        dy[j] = yf[j] - yf[j-1];
        yc[j] = (yf[j] + yf[j-1]) / 2;
    }
    for (int k = 1; k <= Nz; k++) {
        dz[k] = zf[k] - zf[k-1];
        zc[k] = (zf[k] + zf[k-1]) / 2;
    }

    /* Ghost cells */
    dx[0] = dx[1];
    dx[Nx+1] = dx[Nx];
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];
    dz[0] = dz[1];
    dz[Nz+1] = dz[Nz];

    xc[0] = 2*xf[0] - xc[1];
    xc[Nx+1] = 2*xf[Nx] - xc[Nx];
    yc[0] = 2*yf[0] - yc[1];
    yc[Ny+1] = 2*yf[Ny] - yc[Ny];
    zc[0] = 2*zf[0] - zc[1];
    zc[Nz+1] = 2*zf[Nz] - zc[Nz];

    /* Calculate second order derivative coefficients */
    for (int i = 1; i <= Nx; i++) {
        kx_W[i] = dt / (2*Re * (xc[i] - xc[i-1])*dx[i]);
        kx_E[i] = dt / (2*Re * (xc[i+1] - xc[i])*dx[i]);
    }
    for (int j = 1; j <= Ny; j++) {
        ky_S[j] = dt / (2*Re * (yc[j] - yc[j-1])*dy[j]);
        ky_N[j] = dt / (2*Re * (yc[j+1] - yc[j])*dy[j]);
    }
    for (int k = 1; k <= Nz; k++) {
        kz_D[k] = dt / (2*Re * (zc[k] - zc[k-1])*dz[k]);
        kz_U[k] = dt / (2*Re * (zc[k+1] - zc[k])*dz[k]);
    }

    /****** Calculate level set function **************************************/

    Polyhedron_cpt(poly, Nx+2, Ny+2, Nz+2, xc, yc, zc, lvset, 1);

    FILE *fp_lvset = fopen("lvset.txt", "w");
    if (fp_lvset) {
        FOR_INNER_CELL (i, j, k) {
            fprintf(fp_lvset, "%.8lf\n", lvset[i][j][k]);
        }
        fclose(fp_lvset);
    }

    HYPRE_Finalize();
    MPI_Finalize();

#if 0
    /****** Calculate flags ***************************************************/

    /* Level set function is positive           => fluid cell (flag = 1)
       At least one adjacent cell is fluid cell => ghost cell (flag = 2)
       Otherwise                                => solid cell (flag = 0) */
    FOR_ALL_CELL (i, j, k) {
        if (lvset[i][j][k] > 0) {
            flag[i][j][k] = 1;
        }
        else {
            int is_ghost_cell = 0;
            for (int l = 0; l < 6; l++) {
                int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                if (
                    0 <= ni && ni <= Nx+1
                    && 0 <= nj && nj <= Ny+1
                    && 0 <= nk && nk <= Nz+1
                ) {
                    is_ghost_cell |= lvset[ni][nj][nk] > 0;
                }
            }
            flag[i][j][k] = is_ghost_cell ? 2 : 0;
        }
    }

    /****** Calculate number of cells and assign cell ids *********************/

    /* Ghost cells */
    FOR_INNER_CELL (i, j, k) {
        if (flag[i][j][k] == 2) {
            cell_id[i][j][k] = num_tc;
            num_gc++;
            num_tc++;
        }
    }

    /* Fluid cells */
    FOR_ALL_CELL (i, j, k) {
        if (
               (i == 0    && j == 0   )
            || (i == 0    && j == Ny+1)
            || (i == Nx+1 && j == 0   )
            || (i == Nx+1 && j == Ny+1)
            || (i == 0    && k == 0   )
            || (i == 0    && k == Nz+1)
            || (i == Nx+1 && k == 0   )
            || (i == Nx+1 && k == Nz+1)
            || (j == 0    && k == 0   )
            || (j == 0    && k == Nz+1)
            || (j == Ny+1 && k == 0   )
            || (j == Ny+1 && k == Nz+1)
        ) {
            continue;
        }
        if (flag[i][j][k] == 1) {
            cell_id[i][j][k] = num_tc;
            num_tc++;
        }
    }

    /* Calculate cell id. of adjacent cells for all inner cells */
    FOR_INNER_CELL (i, j, k) {
        if (flag[i][j][k] == 1) {
            for (int l = 0; l < 6; l++) {
                int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                adj_cell_id[i][j][k][l] = cell_id[ni][nj][nk];
            }
        }
    }

    /* Print statistics */
    printf("\n");
    printf("# total cells: %d\n", num_tc);
    printf("# ghost cells: %d\n", num_gc);

    /****** Calculate interpolation infos for ghost cells *********************/

    gc_interp_cell_id = calloc(num_gc, sizeof(int [8]));
    gc_interp_coeff = calloc(num_gc, sizeof(double [8]));

    FOR_INNER_CELL (i, j, k) {
        if (flag[i][j][k] == 2) {
            int cur_id = cell_id[i][j][k];

            /* Calculate the gradient of level set function, which is the
               outward unit normal vector to polyhedron face */
            Vector n;
            n.x = (lvset[i+1][j][k] - lvset[i-1][j][k]) / (xc[i+1] - xc[i-1]);
            n.y = (lvset[i][j+1][k] - lvset[i][j-1][k]) / (yc[j+1] - yc[j-1]);
            n.z = k > 0 ?
                (lvset[i][j][k+1] - lvset[i][j][k-1]) / (zc[k+1] - zc[k-1])
                : (lvset[i][j][k+1] - lvset[i][j][k]) / (zc[k+1] - zc[k]);

            /* Calculate the coordinate of mirror point of ghost cell */
            Vector m;
            m.x = xc[i] - 2*lvset[i][j][k]*n.x;
            m.y = yc[j] - 2*lvset[i][j][k]*n.y;
            m.z = zc[k] - 2*lvset[i][j][k]*n.z;

            /* Calculate the indices of 8 adjacent cells of mirror point */
            int iu = upper_bound(Nx+2, xc, m.x);
            int ju = upper_bound(Ny+2, yc, m.y);
            int ku = upper_bound(Nz+2, zc, m.z);

            gc_interp_cell_id[cur_id][0] = cell_id[iu-1][ju-1][ku-1];
            gc_interp_cell_id[cur_id][1] = cell_id[iu-1][ju-1][ku  ];
            gc_interp_cell_id[cur_id][2] = cell_id[iu-1][ju  ][ku-1];
            gc_interp_cell_id[cur_id][3] = cell_id[iu-1][ju  ][ku  ];
            gc_interp_cell_id[cur_id][4] = cell_id[iu  ][ju-1][ku-1];
            gc_interp_cell_id[cur_id][5] = cell_id[iu  ][ju-1][ku  ];
            gc_interp_cell_id[cur_id][6] = cell_id[iu  ][ju  ][ku-1];
            gc_interp_cell_id[cur_id][7] = cell_id[iu  ][ju  ][ku  ];

            /* Calculate the trilinear interpolation coefficients */
            double xl = xc[iu-1], xu = xc[iu];
            double yl = yc[ju-1], yu = yc[ju];
            double zl = zc[ku-1], zu = zc[ku];
            double vol = (xu - xl) * (yu - yl) * (zu - zl);

            gc_interp_coeff[cur_id][0] = (xu-m.x)*(yu-m.y)*(zu-m.z) / vol;
            gc_interp_coeff[cur_id][1] = (xu-m.x)*(yu-m.y)*(m.z-zl) / vol;
            gc_interp_coeff[cur_id][2] = (xu-m.x)*(m.y-yl)*(zu-m.z) / vol;
            gc_interp_coeff[cur_id][3] = (xu-m.x)*(m.y-yl)*(m.z-zl) / vol;
            gc_interp_coeff[cur_id][4] = (m.x-xl)*(yu-m.y)*(zu-m.z) / vol;
            gc_interp_coeff[cur_id][5] = (m.x-xl)*(yu-m.y)*(m.z-zl) / vol;
            gc_interp_coeff[cur_id][6] = (m.x-xl)*(m.y-yl)*(zu-m.z) / vol;
            gc_interp_coeff[cur_id][7] = (m.x-xl)*(m.y-yl)*(m.z-zl) / vol;
        }
    }

    printf("Interpolation done\n");

    /****** Initialize HYPRE variables ****************************************/
    /*===== Matrices for intermediate velocities =============================*/
    /*----- Create and initialize --------------------------------------------*/

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_tc-1, 0, num_tc-1, &A_u1);
    HYPRE_IJMatrixSetObjectType(A_u1, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_u1);

    /*----- Inner cells ------------------------------------------------------*/

    FOR_INNER_CELL (i, j, k) {
        int cur_id = cell_id[i][j][k];

        int ncols;
        int cols[9];
        double values[9];

        /* Fluid cell */
        if (flag[i][j][k] == 1) {
            ncols = 7;
            cols[0] = cur_id;
            for (int l = 0; l < 6; l++) {
                cols[l+1] = adj_cell_id[i][j][k][l];
            }
            values[0] = 1+ky_N[j]+kx_E[i]+ky_S[j]+kx_W[i]+kz_D[k]+kz_U[k];
            values[1] = -ky_N[j];
            values[2] = -kx_E[i];
            values[3] = -ky_S[j];
            values[4] = -kx_W[i];
            values[5] = -kz_D[k];
            values[6] = -kz_U[k];
        }
        /* Ghost cell */
        else if (flag[i][j][k] == 2) {
            int idx = -1;
            for (int l = 0; l < 8; l++) {
                if (gc_interp_cell_id[cur_id][l] == cur_id) {
                    idx = l;
                    break;
                }
            }

            /* If the mirror point is interpolated using the ghost cell
               itself */
            if (idx == -1) {
                ncols = 9;
                cols[0] = cur_id;
                for (int l = 0; l < 8; l++) {
                    cols[l+1] = gc_interp_cell_id[cur_id][l];
                }
                values[0] = 1;
                for (int l = 0; l < 8; l++) {
                    values[l+1] = gc_interp_coeff[cur_id][l];
                }
            }
            /* Otherwise */
            else {
                ncols = 8;
                cols[0] = cur_id;
                values[0] = 1 + gc_interp_coeff[cur_id][idx];
                int cnt = 1;
                for (int l = 0; l < 8; l++) {
                    if (l != idx) {
                        cols[cnt] = gc_interp_cell_id[cur_id][l];
                        values[cnt] = gc_interp_coeff[cur_id][l];
                        cnt++;
                    }
                }
            }
        }

        HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cur_id, cols, values);
    }

    /*----- Outer cells (all fluid cells) ------------------------------------*/

    /* West (i = 0) => velocity inlet; u1[0][j][k] + u1[1][j][k] = 2 */
    for (int j = 1; j <= Ny; j++) {
        for (int k = 1; k <= Nz; k++) {
            int ncols = 2;
            int cols[2] = {cell_id[0][j][k], cell_id[1][j][k]};
            double values[2] = {1, 1};

            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[0][j][k], cols, values);
        }
    }
    /* East (i = Nx+1) => pressure outlet; u1[Nx+1][j][k] is linearly
       extrapolated using u1[Nx-1][j][k] and u1[Nx][j][k] */
    for (int j = 1; j <= Ny; j++) {
        for (int k = 1; k <= Nz; k++) {
            int ncols = 3;
            int cols[3] = {cell_id[Nx+1][j][k], cell_id[Nx][j][k], cell_id[Nx-1][j][k]};
            double values[3] = {xc[Nx] - xc[Nx-1], -(xc[Nx+1] - xc[Nx-1]), xc[Nx+1] - xc[Nx]};

            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[Nx+1][j][k], cols, values);
        }
    }
    /* South (j = 0) => free-slip wall; u1[i][0][k] = u1[i][1][k] */
    for (int i = 1; i <= Nx; i++) {
        for (int k = 1; k <= Ny; k++) {
            int ncols = 2;
            int cols[2] = {cell_id[i][0][k], cell_id[i][1][k]};
            double values[2] = {1, -1};

            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[i][0][k], cols, values);
        }
    }
    /* North (j = Ny+1) => free-slip wall; u1[i][Ny+1][k] = u1[i][Ny][k] */
    for (int i = 1; i <= Nx; i++) {
        for (int k = 1; k <= Ny; k++) {
            int ncols = 2;
            int cols[2] = {cell_id[i][Ny+1][k], cell_id[i][Ny][k]};
            double values[2] = {1, -1};

            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[i][Ny+1][k], cols, values);
        }
    }
    /* Upper (k = Nz+1) => free-slip wall; u1[i][j][Nz+1] = u1[i][j][Nz] */
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            int ncols = 2;
            int cols[2] = {cell_id[i][j][Nz+1], cell_id[i][j][Nz]};
            double values[2] = {1, -1};

            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[i][j][Nz+1], cols, values);
        }
    }

#endif
}
