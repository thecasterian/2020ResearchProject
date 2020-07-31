#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

const int adj[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int bs_upper_bound(int len, double arr[static len], double val) {
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
    /*===== Initialize program and parse arguments ===========================*/
    /* Initialize MPI */
    int myid, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Init();

    /* For the present, the number of process must be 1 */
    if (num_procs != 1) {
        if (myid == 0)
            printf("Must run with 1 processors!\n");
        MPI_Finalize();
        return 0;
    }

    /*===== Read input file ==================================================*/
    /* Input parameters */
    int Nx, Ny;
    double Re, dt;
    int numtstep;

    FILE *fp_in = fopen("cylinder.in", "r");
    if (!fp_in) {
        printf("cannot open file!\n");
        MPI_Finalize();
        return 0;
    }

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

    fprintf(stderr, "Input mesh size: %d x %d\n", Nx, Ny);

    /* Read Re, dt, numtstep */
    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep);

    fclose(fp_in);

    /*===== Define variables =================================================*/
    /* Grid variables */
    double xc[Nx+2], dx[Nx+2];
    double yc[Ny+2], dy[Ny+2];

    /* For IBM */
    double (*const lvset)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    int (*const flag)[Ny+2] = calloc(Nx+2, sizeof(int [Ny+2]));
    int num_tc, num_gc;

    int cell_id[Nx+2][Ny+2];
    int adj_cell_id[Nx+2][Ny+2][4];

    int (*gc_interp_cell_id)[4];
    double (*gc_interp_coeff)[4];

    /* Pressure and velocities */
    double (*const p      )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const p_next )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const p_prime)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));

    double (*const u1      )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const u1_next )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const u1_star )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const u1_tilde)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));

    double (*const u2      )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const u2_next )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const u2_star )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const u2_tilde)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));

    double (*const U1     )[Ny+2] = calloc(Nx+1, sizeof(double [Ny+2]));
    double (*const U1_next)[Ny+2] = calloc(Nx+1, sizeof(double [Ny+2]));
    double (*const U1_star)[Ny+2] = calloc(Nx+1, sizeof(double [Ny+2]));

    double (*const U2     )[Ny+1] = calloc(Nx+2, sizeof(double [Ny+1]));
    double (*const U2_next)[Ny+1] = calloc(Nx+2, sizeof(double [Ny+1]));
    double (*const U2_star)[Ny+1] = calloc(Nx+2, sizeof(double [Ny+1]));

    /* Fluxes */
    double (*const N1     )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const N1_prev)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const N2     )[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const N2_prev)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));

    /* Others */
    double (*const RHS1)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double (*const RHS2)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    double kx_W[Nx+2], kx_E[Nx+2];
    double ky_S[Ny+2], ky_N[Ny+2];

    /* For HYPRE */
    HYPRE_IJMatrix     A_u1, A_u2;
    HYPRE_ParCSRMatrix parcsr_A_u1, parcsr_A_u2;
    HYPRE_IJVector     b_u1, b_u2;
    HYPRE_ParVector    par_b_u1, par_b_u2;
    HYPRE_IJVector     x_u1, x_u2;
    HYPRE_ParVector    par_x_u1, par_x_u2;

    HYPRE_IJMatrix     A_p;
    HYPRE_ParCSRMatrix parcsr_A_p;
    HYPRE_IJVector     b_p;
    HYPRE_ParVector    par_b_p;
    HYPRE_IJVector     x_p;
    HYPRE_ParVector    par_x_p;

    HYPRE_Solver solver_u1, solver_u2, solver_p, precond;

    int *vector_rows;
    double *vector_values_u1, *vector_values_u2, *vector_values_p;
    double *vector_zeros, *vector_res;

    double final_res_norm;

    /* Stream function */
    double (*const psi)[Ny+1] = calloc(Nx+1, sizeof(double [Ny+1]));

    /*===== Calculate grid variables =========================================*/
    for (int i = 1; i <= Nx; i++) {
        dx[i] = xf[i] - xf[i-1];
        xc[i] = (xf[i] + xf[i-1]) / 2;
    }
    for (int j = 1; j <= Ny; j++) {
        dy[j] = yf[j] - yf[j-1];
        yc[j] = (yf[j] + yf[j-1]) / 2;
    }

    /* Ghost cells */
    dx[0] = dx[1];
    dx[Nx+1] = dx[Nx];
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];
    xc[0] = 2*xf[0] - xc[1];
    xc[Nx+1] = 2*xf[Nx] - xc[Nx];
    yc[0] = 2*yf[0] - yc[1];
    yc[Ny+1] = 2*yf[Ny] - yc[Ny];

    /* Calculate second order derivative coefficients */
    for (int i = 1; i <= Nx; i++) {
        kx_W[i] = dt / (2*Re * (xc[i] - xc[i-1])*dx[i]);
        kx_E[i] = dt / (2*Re * (xc[i+1] - xc[i])*dx[i]);
    }
    for (int j = 1; j <= Ny; j++) {
        ky_S[j] = dt / (2*Re * (yc[j] - yc[j-1])*dy[j]);
        ky_N[j] = dt / (2*Re * (yc[j+1] - yc[j])*dy[j]);
    }

    /*===== Calculate level set function and flags ===========================*/
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            lvset[i][j] = sqrt(xc[i]*xc[i] + yc[j]*yc[j]) - 0.5;
        }
    }

    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            if (lvset[i][j] > 0) {
                flag[i][j] = 1;
            }
            else {
                bool is_ghost_cell = false;
                for (int k = 0; k < 4; k++) {
                    int ni = i + adj[k][0], nj = j + adj[k][1];
                    if (0 <= ni && ni <= Nx+1 && 0 <= j && j <= Ny+1) {
                        is_ghost_cell |= lvset[ni][nj] > 0;
                    }
                }
                flag[i][j] = is_ghost_cell ? 2 : 0;
            }
        }
    }

    /*===== Assign cell ids ==================================================*/
    num_tc = num_gc = 0;
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            cell_id[i][j] = 0;
        }
    }

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            if (flag[i][j] == 2) {
                num_gc++;
                num_tc++;
                cell_id[i][j] = num_tc;
            }
        }
    }
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            if (
                (i == 0 && j == 0)
                || (i == 0 && j == Ny+1)
                || (i == Nx+1 && j == 0)
                || (i == Nx+1 && j == Ny+1)
            ) {
                continue;
            }
            if (flag[i][j] == 1) {
                num_tc++;
                cell_id[i][j] = num_tc;
            }
        }
    }

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            if (flag[i][j] == 1) {
                for (int k = 0; k < 4; k++) {
                    int ni = i + adj[k][0], nj = j + adj[k][1];
                    adj_cell_id[i][j][k] = cell_id[ni][nj];
                }
            }
        }
    }

    fprintf(stderr, "# total cells: %d\n", num_tc);
    fprintf(stderr, "# ghost cells: %d\n", num_gc);

    /*===== Calculate interpolation infos for ghost cells ====================*/
    gc_interp_cell_id = calloc(num_gc+1, sizeof(int [4]));
    gc_interp_coeff = calloc(num_gc+1, sizeof(double [4]));

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            if (flag[i][j] == 2) {
                const int cur_id = cell_id[i][j];

                const double l = sqrt(xc[i]*xc[i] + yc[j]*yc[j]);
                const double nx = xc[i] / l, ny = yc[j] / l;
                const double mx = xc[i] + 2*(0.5-l)*nx, my = yc[j] + 2*(0.5-l)*ny;

                const int iright = bs_upper_bound(Nx+2, xc, mx);
                const int jupper = bs_upper_bound(Ny+2, yc, my);

                gc_interp_cell_id[cur_id][0] = cell_id[iright-1][jupper-1];
                gc_interp_cell_id[cur_id][1] = cell_id[iright][jupper-1];
                gc_interp_cell_id[cur_id][2] = cell_id[iright-1][jupper];
                gc_interp_cell_id[cur_id][3] = cell_id[iright][jupper];

                const double x1 = xc[iright-1], x2 = xc[iright];
                const double y1 = yc[jupper-1], y2 = yc[jupper];

                const double area = (y2 - y1) * (x2 - x1);
                gc_interp_coeff[cur_id][0] = (y2 - my) * (x2 - mx) / area;
                gc_interp_coeff[cur_id][1] = (y2 - my) * (mx - x1) / area;
                gc_interp_coeff[cur_id][2] = (my - y1) * (x2 - mx) / area;
                gc_interp_coeff[cur_id][3] = (my - y1) * (mx - x1) / area;
            }
        }
    }

    fprintf(stderr, "interpolation done\n");

    /*===== Initialize HYPRE variables =======================================*/
    /* Define matrices */
    {
        /* Matrix for intermediate velocity */
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 1, num_tc, 1, num_tc, &A_u1);
        HYPRE_IJMatrixSetObjectType(A_u1, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_u1);

        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                int ncols;
                int cols[5];
                double values[5];

                if (flag[i][j] == 1) {
                    ncols = 5;
                    cols[0] = cell_id[i][j];
                    for (int k = 0; k < 4; k++) {
                        cols[k+1] = adj_cell_id[i][j][k];
                    }
                    values[0] = 1 + ky_N[j] + kx_E[i] + ky_S[j] + kx_W[i];
                    values[1] = -ky_N[j];
                    values[2] = -kx_E[i];
                    values[3] = -ky_S[j];
                    values[4] = -kx_W[i];
                }
                else if (flag[i][j] == 2) {
                    int idx = -1;
                    for (int k = 0; k < 4; k++) {
                        if (gc_interp_cell_id[cell_id[i][j]][k] == cell_id[i][j]) {
                            idx = k;
                            break;
                        }
                    }

                    if (idx == -1) {
                        ncols = 5;
                        cols[0] = cell_id[i][j];
                        for (int k = 0; k < 4; k++) {
                            cols[k+1] = gc_interp_cell_id[cell_id[i][j]][k];
                        }
                        values[0] = 1;
                        for (int k = 0; k < 4; k++) {
                            values[k+1] = gc_interp_coeff[cell_id[i][j]][k];
                        }
                    }
                    else {
                        ncols = 4;
                        cols[0] = cell_id[i][j];
                        int cnt = 1;
                        for (int k = 0; k < 4; k++) {
                            if (k != idx) {
                                cols[cnt++] = gc_interp_cell_id[cell_id[i][j]][k];
                            }
                        }
                        values[0] = 1 + gc_interp_coeff[cell_id[i][j]][idx];
                        cnt = 1;
                        for (int k = 0; k < 4; k++) {
                            if (k != idx) {
                                values[cnt++] = gc_interp_coeff[cell_id[i][j]][k];
                            }
                        }
                    }
                }

                HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[i][j], cols, values);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            int ncols = 2;
            int cols[2];
            double values[2] = {1, 1};

            /* j = 0; u1[i][0] + u[i][1] = 2 */
            cols[0] = cell_id[i][0];
            cols[1] = cell_id[i][1];
            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[i][0], cols, values);

            /* j = Ny+1; u1[i][Ny+1] + u1[i][Ny] = 2 */
            cols[0] = cell_id[i][Ny+1];
            cols[1] = cell_id[i][Ny];
            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[i][Ny+1], cols, values);
        }
        for (int j = 1; j <= Ny; j++) {
            int ncols = 2;
            int cols[2];
            double values[2] = {1, 1};

            /* i = 0; u1[0][j] + u1[1][j] = 2 */
            cols[0] = cell_id[0][j];
            cols[1] = cell_id[1][j];
            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[0][j], cols, values);

            /* i = Nx+1; u1[Nx+1][j] - u1[Nx][j] = 0 */
            cols[0] = cell_id[Nx+1][j];
            cols[1] = cell_id[Nx][j];
            // values[1] = -1;
            HYPRE_IJMatrixSetValues(A_u1, 1, &ncols, &cell_id[Nx+1][j], cols, values);
        }

        HYPRE_IJMatrixAssemble(A_u1);
        HYPRE_IJMatrixGetObject(A_u1, (void **)&parcsr_A_u1);

        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 1, num_tc, 1, num_tc, &A_u2);
        HYPRE_IJMatrixSetObjectType(A_u2, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_u2);

        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                int ncols;
                int cols[5];
                double values[5];

                if (flag[i][j] == 1) {
                    ncols = 5;
                    cols[0] = cell_id[i][j];
                    for (int k = 0; k < 4; k++) {
                        cols[k+1] = adj_cell_id[i][j][k];
                    }
                    values[0] = 1 + ky_N[j] + kx_E[i] + ky_S[j] + kx_W[i];
                    values[1] = -ky_N[j];
                    values[2] = -kx_E[i];
                    values[3] = -ky_S[j];
                    values[4] = -kx_W[i];
                }
                else if (flag[i][j] == 2) {
                    int idx = -1;
                    for (int k = 0; k < 4; k++) {
                        if (gc_interp_cell_id[cell_id[i][j]][k] == cell_id[i][j]) {
                            idx = k;
                            break;
                        }
                    }

                    if (idx == -1) {
                        ncols = 5;
                        cols[0] = cell_id[i][j];
                        for (int k = 0; k < 4; k++) {
                            cols[k+1] = gc_interp_cell_id[cell_id[i][j]][k];
                        }
                        values[0] = 1;
                        for (int k = 0; k < 4; k++) {
                            values[k+1] = gc_interp_coeff[cell_id[i][j]][k];
                        }
                    }
                    else {
                        ncols = 4;
                        cols[0] = cell_id[i][j];
                        int cnt = 1;
                        for (int k = 0; k < 4; k++) {
                            if (k != idx) {
                                cols[cnt++] = gc_interp_cell_id[cell_id[i][j]][k];
                            }
                        }
                        values[0] = 1 + gc_interp_coeff[cell_id[i][j]][idx];
                        cnt = 1;
                        for (int k = 0; k < 4; k++) {
                            if (k != idx) {
                                values[cnt++] = gc_interp_coeff[cell_id[i][j]][k];
                            }
                        }
                    }
                }

                HYPRE_IJMatrixSetValues(A_u2, 1, &ncols, &cell_id[i][j], cols, values);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            int ncols = 2;
            int cols[2];
            double values[2] = {1, 1};

            /* j = 0; u2[i][0] + u2[i][1] = 0 */
            cols[0] = cell_id[i][0];
            cols[1] = cell_id[i][1];
            HYPRE_IJMatrixSetValues(A_u2, 1, &ncols, &cell_id[i][0], cols, values);

            /* j = Ny+1; u2[i][Ny+1] + u2[i][Ny] = 0 */
            cols[0] = cell_id[i][Ny+1];
            cols[1] = cell_id[i][Ny];
            HYPRE_IJMatrixSetValues(A_u2, 1, &ncols, &cell_id[i][Ny+1], cols, values);
        }
        for (int j = 1; j <= Ny; j++) {
            int ncols = 2;
            int cols[2];
            double values[2] = {1, 1};

            /* i = 0; u2[0][j] + u2[1][j] = 0 */
            cols[0] = cell_id[0][j];
            cols[1] = cell_id[1][j];
            HYPRE_IJMatrixSetValues(A_u2, 1, &ncols, &cell_id[0][j], cols, values);

            /* i = Nx+1; u2[Nx+1][j] - u2[Nx][j] = 0 */
            cols[0] = cell_id[Nx+1][j];
            cols[1] = cell_id[Nx][j];
            // values[1] = -1;
            HYPRE_IJMatrixSetValues(A_u2, 1, &ncols, &cell_id[Nx+1][j], cols, values);
        }

        HYPRE_IJMatrixAssemble(A_u2);
        HYPRE_IJMatrixGetObject(A_u2, (void **)&parcsr_A_u2);

        /* Matrix for pressure */
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 1, num_tc, 1, num_tc, &A_p);
        HYPRE_IJMatrixSetObjectType(A_p, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_p);

        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                if (i == 1 & j == 1) {
                    int ncols = 1;
                    int cols[1] = {cell_id[1][1]};
                    double values[1] = {1};
                    HYPRE_IJMatrixSetValues(A_p, 1, &ncols, &cell_id[i][j], cols, values);
                    continue;
                }

                int ncols;
                int cols[5];
                double values[5];

                if (flag[i][j] == 1) {
                    ncols = 5;
                    cols[0] = cell_id[i][j];
                    for (int k = 0; k < 4; k++) {
                        cols[k+1] = adj_cell_id[i][j][k];
                    }
                    values[0] = -(ky_N[j] + kx_E[i] + ky_S[j] + kx_W[i]);
                    values[1] = ky_N[j];
                    values[2] = kx_E[i];
                    values[3] = ky_S[j];
                    values[4] = kx_W[i];
                }
                else if (flag[i][j] == 2) {
                    int idx = -1;
                    for (int k = 0; k < 4; k++) {
                        if (gc_interp_cell_id[cell_id[i][j]][k] == cell_id[i][j]) {
                            idx = k;
                            break;
                        }
                    }

                    if (idx == -1) {
                        ncols = 5;
                        cols[0] = cell_id[i][j];
                        for (int k = 0; k < 4; k++) {
                            cols[k+1] = gc_interp_cell_id[cell_id[i][j]][k];
                        }
                        values[0] = 1;
                        for (int k = 0; k < 4; k++) {
                            values[k+1] = -gc_interp_coeff[cell_id[i][j]][k];
                        }
                    }
                    else {
                        ncols = 4;
                        cols[0] = cell_id[i][j];
                        int cnt = 1;
                        for (int k = 0; k < 4; k++) {
                            if (k != idx) {
                                cols[cnt++] = gc_interp_cell_id[cell_id[i][j]][k];
                            }
                        }
                        values[0] = 1 - gc_interp_coeff[cell_id[i][j]][idx];
                        cnt = 1;
                        for (int k = 0; k < 4; k++) {
                            if (k != idx) {
                                values[cnt++] = -gc_interp_coeff[cell_id[i][j]][k];
                            }
                        }
                    }
                }
                HYPRE_IJMatrixSetValues(A_p, 1, &ncols, &cell_id[i][j], cols, values);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            int ncols = 2;
            int cols[2];
            double values[2] = {1, -1};

            /* j = 0; p[i][0] - p[i][1] = 0 */
            cols[0] = cell_id[i][0];
            cols[1] = cell_id[i][1];
            HYPRE_IJMatrixSetValues(A_p, 1, &ncols, &cell_id[i][0], cols, values);

            /* j = Ny+1; p[i][Ny+1] - p[i][Ny] = 0 */
            cols[0] = cell_id[i][Ny+1];
            cols[1] = cell_id[i][Ny];
            HYPRE_IJMatrixSetValues(A_p, 1, &ncols, &cell_id[i][Ny+1], cols, values);
        }
        for (int j = 1; j <= Ny; j++) {
            int ncols;
            int cols[3];
            double values[3];

            /* i = 0; p[0][j] - p[1][j] = 0 */
            ncols = 2;
            cols[0] = cell_id[0][j];
            cols[1] = cell_id[1][j];
            values[0] = 1;
            values[1] = -1;
            HYPRE_IJMatrixSetValues(A_p, 1, &ncols, &cell_id[0][j], cols, values);

            /* i = Nx+1; p[Nx+1][j] - 2*p[Nx][j] + p[Nx-1][j] = 0 */
            // ncols = 3;
            // cols[0] = cell_id[Nx+1][j];
            // cols[1] = cell_id[Nx][j];
            // cols[2] = cell_id[Nx-1][j];
            // values[0] = 1;
            // values[1] = -2;
            // values[2] = 1;
            ncols = 2;
            cols[0] = cell_id[Nx+1][j];
            cols[1] = cell_id[Nx][j];
            values[0] = 1;
            values[1] = -1;
            HYPRE_IJMatrixSetValues(A_p, 1, &ncols, &cell_id[Nx+1][j], cols, values);
        }

        HYPRE_IJMatrixAssemble(A_p);
        HYPRE_IJMatrixGetObject(A_p, (void **)&parcsr_A_p);
    }

    /* Initialize RHS vector */
    {
        /* RHS vector */
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, num_tc, &b_u1);
        HYPRE_IJVectorSetObjectType(b_u1, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b_u1);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, num_tc, &b_u2);
        HYPRE_IJVectorSetObjectType(b_u2, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b_u2);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, num_tc, &b_p);
        HYPRE_IJVectorSetObjectType(b_p, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b_p);

        /* Solution vector */
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, num_tc, &x_u1);
        HYPRE_IJVectorSetObjectType(x_u1, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x_u1);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, num_tc, &x_u2);
        HYPRE_IJVectorSetObjectType(x_u2, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x_u2);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 1, num_tc, &x_p);
        HYPRE_IJVectorSetObjectType(x_p, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x_p);
    }

    /* Initialize solver */
    {
        HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_u1);
        HYPRE_BiCGSTABSetMaxIter(solver_u1, 1000);
        HYPRE_BiCGSTABSetTol(solver_u1, 1e-12);

        HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_u2);
        HYPRE_BiCGSTABSetMaxIter(solver_u2, 1000);
        HYPRE_BiCGSTABSetTol(solver_u2, 1e-12);

        HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_p);
        HYPRE_BiCGSTABSetMaxIter(solver_p, 1000);
        HYPRE_BiCGSTABSetTol(solver_p, 1e-12);

        HYPRE_BoomerAMGCreate(&precond);
        HYPRE_BoomerAMGSetCoarsenType(precond, 6);
        HYPRE_BoomerAMGSetOldDefault(precond);
        HYPRE_BoomerAMGSetRelaxType(precond, 6);
        HYPRE_BoomerAMGSetNumSweeps(precond, 1);
        HYPRE_BoomerAMGSetTol(precond, 0);
        HYPRE_BoomerAMGSetMaxIter(precond, 1);

        HYPRE_BiCGSTABSetPrecond(solver_u1, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                                 (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);
        HYPRE_BiCGSTABSetPrecond(solver_u2, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                                 (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);
        HYPRE_BiCGSTABSetPrecond(solver_p, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                                 (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);
    }

    vector_rows = calloc(num_tc, sizeof(int));
    vector_values_u1 = calloc(num_tc, sizeof(double));
    vector_values_u2 = calloc(num_tc, sizeof(double));
    vector_values_p = calloc(num_tc, sizeof(double));
    vector_zeros = calloc(num_tc, sizeof(double));
    vector_res = calloc(num_tc, sizeof(double));

    for (int i = 0; i < num_tc; i++) {
        vector_rows[i] = i + 1;
        vector_values_u1[i] = 0;
        vector_values_u2[i] = 0;
        vector_values_p[i] = 0;
        vector_zeros[i] = 0;
    }

    for (int i = 1; i <= Nx; i++) {
        vector_values_u1[cell_id[i][0]-1] = 2;
        vector_values_u1[cell_id[i][Ny+1]-1] = 2;
    }
    for (int j = 1; j <= Ny; j++) {
        vector_values_u1[cell_id[0][j]-1] = 2;
        vector_values_u1[cell_id[Nx+1][j]-1] = 2;
    }

    fprintf(stderr, "HYPRE done\n");

    /*===== Initialize flow ==================================================*/
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            u1[i][j] = 1;
        }
    }
    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            U1[i][j] = 1;
        }
    }

    // memcpy(u1_next, u1, sizeof(double)*(Nx+2)*(Ny+2));
    // memcpy(U1_next, U1, sizeof(double)*(Nx+1)*(Ny+2));
    // memcpy(U1_star, U1, sizeof(double)*(Nx+1)*(Ny+2));

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            double u1_e = (u1[i][j] * dx[i+1] + u1[i+1][j] * dx[i]) / (dx[i] + dx[i+1]);
            double u2_e = (u2[i][j] * dx[i+1] + u2[i+1][j] * dx[i]) / (dx[i] + dx[i+1]);
            double u1_w = (u1[i-1][j] * dx[i] + u1[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
            double u2_w = (u2[i-1][j] * dx[i] + u2[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
            double u1_n = (u1[i][j] * dy[j+1] + u1[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
            double u2_n = (u2[i][j] * dy[j+1] + u2[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
            double u1_s = (u1[i][j-1] * dy[j] + u1[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);
            double u2_s = (u2[i][j-1] * dy[j] + u2[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);

            N1_prev[i][j] = (U1[i][j]*u1_e - U1[i-1][j]*u1_w) / dx[i]
                + (U2[i][j]*u1_n - U2[i][j-1]*u1_s) / dy[j];
            N2_prev[i][j] = (U1[i][j]*u2_e - U1[i-1][j]*u2_w) / dx[i]
                + (U2[i][j]*u2_n - U2[i][j-1]*u2_s) / dy[j];
        }
    }

    /*===== Main loop ========================================================*/
    for (int tstep = 1; tstep <= numtstep; tstep++) {
        /* Calculate N */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u1_e = (u1[i][j] * dx[i+1] + u1[i+1][j] * dx[i]) / (dx[i] + dx[i+1]);
                double u2_e = (u2[i][j] * dx[i+1] + u2[i+1][j] * dx[i]) / (dx[i] + dx[i+1]);
                double u1_w = (u1[i-1][j] * dx[i] + u1[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
                double u2_w = (u2[i-1][j] * dx[i] + u2[i][j] * dx[i-1]) / (dx[i-1] + dx[i]);
                double u1_n = (u1[i][j] * dy[j+1] + u1[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
                double u2_n = (u2[i][j] * dy[j+1] + u2[i][j+1] * dy[j]) / (dy[j] + dy[j+1]);
                double u1_s = (u1[i][j-1] * dy[j] + u1[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);
                double u2_s = (u2[i][j-1] * dy[j] + u2[i][j] * dy[j-1]) / (dy[j-1] + dy[j]);

                N1[i][j] = (U1[i][j]*u1_e - U1[i-1][j]*u1_w) / dx[i]
                    + (U2[i][j]*u1_n - U2[i][j-1]*u1_s) / dy[j];
                N2[i][j] = (U1[i][j]*u2_e - U1[i-1][j]*u2_w) / dx[i]
                    + (U2[i][j]*u2_n - U2[i][j-1]*u2_s) / dy[j];
            }
        }

        FILE *fp_N = fopen("N1.txt", "w");
        if (fp_N) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_N, "%17.14lf ", N1[i][j]);
                }
                fprintf(fp_N, "\n");
            }
            fclose(fp_N);
        }
        fp_N = fopen("N2.txt", "w");
        if (fp_N) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_N, "%17.14lf ", N2[i][j]);
                }
                fprintf(fp_N, "\n");
            }
            fclose(fp_N);
        }

        /* Calculate RHS */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                if (flag[i][j] == 1) {
                    RHS1[i][j] = -dt/2 * (3*N1[i][j] - N1_prev[i][j])
                        - dt * (p[i+1][j] - p[i-1][j]) / (xc[i+1] - xc[i-1])
                        + (1 - ky_N[j] - kx_E[i] - ky_S[j] - kx_W[i])*u1[i][j]
                        + ky_N[j]*u1[i][j+1] + kx_E[i]*u1[i+1][j]
                        + ky_S[j]*u1[i][j-1] + kx_W[i]*u1[i-1][j];
                    RHS2[i][j] = -dt/2 * (3*N2[i][j] - N2_prev[i][j])
                        - dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1])
                        + (1 - ky_N[j] - kx_E[i] - ky_S[j] - kx_W[i])*u2[i][j]
                        + ky_N[j]*u2[i][j+1] + kx_E[i]*u2[i+1][j]
                        + ky_S[j]*u2[i][j-1] + kx_W[i]*u2[i-1][j];
                }
            }
        }

        FILE *fp_RHS = fopen("RHS1.txt", "w");
        if (fp_RHS) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_RHS, "%17.14lf ", RHS1[i][j]);
                }
                fprintf(fp_RHS, "\n");
            }
            fclose(fp_RHS);
        }
        fp_RHS = fopen("RHS2.txt", "w");
        if (fp_RHS) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_RHS, "%17.14lf ", RHS2[i][j]);
                }
                fprintf(fp_RHS, "\n");
            }
            fclose(fp_RHS);
        }

        /* Calculate u_star */
        {
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (flag[i][j] == 1) {
                        vector_values_u1[cell_id[i][j]-1] = RHS1[i][j];
                    }
                }
            }
            HYPRE_IJVectorSetValues(b_u1, num_tc, vector_rows, vector_values_u1);

            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (flag[i][j] == 1) {
                        vector_values_u2[cell_id[i][j]-1] = RHS2[i][j];
                    }
                }
            }
            HYPRE_IJVectorSetValues(b_u2, num_tc, vector_rows, vector_values_u2);

            HYPRE_IJVectorSetValues(x_u1, num_tc, vector_rows, vector_zeros);
            HYPRE_IJVectorSetValues(x_u2, num_tc, vector_rows, vector_zeros);

            HYPRE_IJVectorAssemble(b_u1);
            HYPRE_IJVectorAssemble(b_u2);
            HYPRE_IJVectorAssemble(x_u1);
            HYPRE_IJVectorAssemble(x_u2);

            HYPRE_IJVectorGetObject(b_u1, (void **)&par_b_u1);
            HYPRE_IJVectorGetObject(b_u2, (void **)&par_b_u2);
            HYPRE_IJVectorGetObject(x_u1, (void **)&par_x_u1);
            HYPRE_IJVectorGetObject(x_u2, (void **)&par_x_u2);

            HYPRE_ParCSRBiCGSTABSetup(solver_u1, parcsr_A_u1, par_b_u1, par_x_u1);
            HYPRE_ParCSRBiCGSTABSetup(solver_u2, parcsr_A_u2, par_b_u2, par_x_u2);

            HYPRE_ParCSRBiCGSTABSolve(solver_u1, parcsr_A_u1, par_b_u1, par_x_u1);
            HYPRE_ParCSRBiCGSTABSolve(solver_u2, parcsr_A_u2, par_b_u2, par_x_u2);

            HYPRE_IJVectorGetValues(x_u1, num_tc, vector_rows, vector_res);
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    if (cell_id[i][j]) {
                        u1_star[i][j] = vector_res[cell_id[i][j]-1];
                    }
                }
            }

            HYPRE_IJVectorGetValues(x_u2, num_tc, vector_rows, vector_res);
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    if (cell_id[i][j]) {
                        u2_star[i][j] = vector_res[cell_id[i][j]-1];
                    }
                }
            }
        }

        FILE *fp_ustar = fopen("u1_star.txt", "w");
        if (fp_ustar) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_ustar, "%17.14lf ", u1_star[i][j]);
                }
                fprintf(fp_ustar, "\n");
            }
            fclose(fp_ustar);
        }
        fp_ustar = fopen("u2_star.txt", "w");
        if (fp_ustar) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_ustar, "%17.14lf ", u2_star[i][j]);
                }
                fprintf(fp_ustar, "\n");
            }
            fclose(fp_ustar);
        }

        /* Calculate u_tilde */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                if (flag[i][j] == 1) {
                    u1_tilde[i][j] = u1_star[i][j] + dt * (p[i+1][j] - p[i-1][j]) / (xc[i+1] - xc[i-1]);
                    u2_tilde[i][j] = u2_star[i][j] + dt * (p[i][j+1] - p[i][j-1]) / (yc[j+1] - yc[j-1]);
                }
            }
        }
        for (int i = 1; i <= Nx; i++) {
            u1_tilde[i][0] = u1_star[i][0] + dt * (p[i+1][0] - p[i-1][0]) / (xc[i+1] - xc[i-1]);
            u1_tilde[i][Ny+1] = u1_star[i][Ny+1] + dt * (p[i+1][Ny+1] - p[i-1][Ny+1]) / (xc[i+1] - xc[i-1]);
            u2_tilde[i][0] = u2_star[i][0] + dt * (p[i][1] - p[i][0]) / (yc[1] - yc[0]);
            u2_tilde[i][Ny+1] = u2_star[i][Ny+1] + dt * (p[i][Ny+1] - p[i][Ny]) / (yc[Ny+1] - yc[Ny]);
        }
        for (int j = 1; j <= Ny; j++) {
            u1_tilde[0][j] = u1_star[0][j] + dt * (p[1][j] - p[0][j]) / (xc[1] - xc[0]);
            u1_tilde[Nx+1][j] = u1_star[Nx+1][j] + dt * (p[Nx+1][j] - p[Nx][j]) / (xc[Nx+1] - xc[Nx]);
            u2_tilde[0][j] = u2_star[0][j] + dt * (p[0][j+1] - p[0][j-1]) / (yc[j+1] - yc[j-1]);
            u2_tilde[Nx+1][j] = u2_star[Nx+1][j] + dt * (p[Nx+1][j+1] - p[Nx+1][j-1]) / (yc[j+1] - yc[j-1]);
        }

        FILE *fp_utilde = fopen("u1_tilde.txt", "w");
        if (fp_utilde) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_utilde, "%17.14lf ", u1_tilde[i][j]);
                }
                fprintf(fp_utilde, "\n");
            }
            fclose(fp_utilde);
        }
        fp_utilde = fopen("u2_tilde.txt", "w");
        if (fp_utilde) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_utilde, "%17.14lf ", u2_tilde[i][j]);
                }
                fprintf(fp_utilde, "\n");
            }
            fclose(fp_utilde);
        }

        /* Calculate U_star */
        for (int i = 0; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_star[i][j] = (u1_tilde[i][j]*dx[i+1] + u1_tilde[i+1][j]*dx[i]) / (dx[i] + dx[i+1])
                    - dt * (p[i+1][j] - p[i][j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                U2_star[i][j] = (u2_tilde[i][j]*dy[j+1] + u2_tilde[i][j+1]*dy[j]) / (dy[j] + dy[j+1])
                    - dt * (p[i][j+1] - p[i][j]) / (yc[j+1] - yc[j]);
            }
        }

        FILE *fp_Ustar = fopen("U1_star.txt", "w");
        if (fp_Ustar) {
            for (int i = 0; i <= Nx; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_Ustar, "%17.14lf ", U1_star[i][j]);
                }
                fprintf(fp_Ustar, "\n");
            }
            fclose(fp_Ustar);
        }
        fp_Ustar = fopen("U2_star.txt", "w");
        if (fp_Ustar) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny; j++) {
                    fprintf(fp_Ustar, "%17.14lf ", U2_star[i][j]);
                }
                fprintf(fp_Ustar, "\n");
            }
            fclose(fp_Ustar);
        }

        FILE *fp_Q = fopen("Q.txt", "w");
        if (fp_Q) {
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    double Q = 1 / (2*Re) * (
                        (U1_star[i][j] - U1_star[i-1][j]) / dx[i]
                        + (U2_star[i][j] - U2_star[i][j-1]) / dy[j]
                    );
                    fprintf(fp_Q, "%17.14lf ", flag[i][j] == 1 ? Q : 0);
                }
                fprintf(fp_Q, "\n");
            }
            fclose(fp_Q);
        }

        /* Calculate p_prime */
        {
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (flag[i][j] == 1) {
                        vector_values_p[cell_id[i][j]-1] = 1 / (2*Re) * (
                            (U1_star[i][j] - U1_star[i-1][j]) / dx[i]
                            + (U2_star[i][j] - U2_star[i][j-1]) / dy[j]
                        );
                    }
                }
            }
            HYPRE_IJVectorSetValues(b_p, num_tc, vector_rows, vector_values_p);

            HYPRE_IJVectorSetValues(x_p, num_tc, vector_rows, vector_zeros);

            HYPRE_IJVectorAssemble(b_p);
            HYPRE_IJVectorAssemble(x_p);

            HYPRE_IJVectorGetObject(b_p, (void **)&par_b_p);
            HYPRE_IJVectorGetObject(x_p, (void **)&par_x_p);

            HYPRE_ParCSRBiCGSTABSetup(solver_p, parcsr_A_p, par_b_p, par_x_p);

            HYPRE_ParCSRBiCGSTABSolve(solver_p, parcsr_A_p, par_b_p, par_x_p);

            HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver_p, &final_res_norm);
            if (final_res_norm > 1e-6) {
                fprintf(stderr, "not converged! \n");
            }
            fprintf(stderr, "final residual: %.14lf\n", final_res_norm);

            HYPRE_IJVectorGetValues(x_p, num_tc, vector_rows, vector_res);
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    if (cell_id[i][j]) {
                        p_prime[i][j] = vector_res[cell_id[i][j]-1];
                    }
                }
            }
        }

        FILE *fp_pprime = fopen("pprime.txt", "w");
        if (fp_pprime) {
            for (int i = 0; i <= Nx+1; i++) {
                for (int j = 0; j <= Ny+1; j++) {
                    fprintf(fp_pprime, "%20.14lf ", p_prime[i][j]);
                }
                fprintf(fp_pprime, "\n");
            }
        }

        /* Calculate p_next */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                if (cell_id[i][j]) {
                    p_next[i][j] = p[i][j] + p_prime[i][j];
                }
            }
        }

        /* Calculate u_next */
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                if (cell_id[i][j]) {
                    u1_next[i][j] = u1_star[i][j] - dt * (p_prime[i+1][j] - p_prime[i-1][j]) / (xc[i+1] - xc[i-1]);
                    u2_next[i][j] = u2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j-1]) / (yc[j+1] - yc[j-1]);
                }
            }
        }

        /* Calculate U_next */
        for (int i = 0; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                U1_next[i][j] = U1_star[i][j] - dt * (p_prime[i+1][j] - p_prime[i][j]) / (xc[i+1] - xc[i]);
            }
        }
        for (int i = 1; i <= Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                U2_next[i][j] = U2_star[i][j] - dt * (p_prime[i][j+1] - p_prime[i][j]) / (yc[j+1] - yc[j]);
            }
        }

        /* Set velocity boundary conditions */
        for (int i = 1; i <= Nx; i++) {
            u1_next[i][0] = 2 - u1_next[i][1];
            u1_next[i][Ny+1] = 2 - u1_next[i][Ny];
            u2_next[i][0] = -u2_next[i][1];
            u2_next[i][Ny+1] = -u2_next[i][Ny];
        }
        for (int j = 1; j <= Ny; j++) {
            u1_next[0][j] = 2 - u1_next[Nx][j];
            u1_next[Nx+1][j] = 2 - u1_next[1][j];
            u2_next[0][j] = -u2_next[Nx][j];
            u2_next[Nx+1][j] = -u2_next[1][j];
        }
        for (int i = 0; i <= Nx; i++) {
            U1_next[i][0] = 2 - U1_next[i][1];
            U1_next[i][Ny+1] = 2 - U1_next[i][Ny];
        }
        for (int j = 0; j <= Ny; j++) {
            U2_next[0][j] = -U2_next[Nx][j];
            U2_next[Nx+1][j] = -U2_next[1][j];
        }

        /* Update for next time step */
        memcpy(p, p_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(u1, u1_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(u2, u2_next, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(U1, U1_next, sizeof(double)*(Nx+1)*(Ny+2));
        memcpy(U2, U2_next, sizeof(double)*(Nx+2)*(Ny+1));
        memcpy(N1_prev, N1, sizeof(double)*(Nx+2)*(Ny+2));
        memcpy(N2_prev, N2, sizeof(double)*(Nx+2)*(Ny+2));

        if (tstep % 50 == 0) {
            printf("tstep: %d\n", tstep);
        }
    }

    /*===== Export result ====================================================*/
    /* Calculate streamfunction */
    for (int i = 1; i <= Nx; i++) {
        psi[i][0] = psi[i-1][0] - dx[i] * U2[i][0];
    }
    for (int i = 0; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            psi[i][j] = psi[i][j-1] + dy[j] * U1[i][j];
        }
    }

    /* Write to output file */
    FILE *fp_out = fopen("cylinder_result.txt", "w");
    if (fp_out) {
        for (int i = 0; i <= Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                fprintf(fp_out, "%.14lf ", psi[i][j]);
            }
            fprintf(fp_out, "\n");
        }
        fclose(fp_out);
    }

    fp_out = fopen("u1.txt", "w");
    if (fp_out) {
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                fprintf(fp_out, "%17.14lf ", u1[i][j]);
            }
            fprintf(fp_out, "\n");
        }
        fclose(fp_out);
    }
    fp_out = fopen("u2.txt", "w");
    if (fp_out) {
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                fprintf(fp_out, "%17.14lf ", u2[i][j]);
            }
            fprintf(fp_out, "\n");
        }
        fclose(fp_out);
    }
    fp_out = fopen("p.txt", "w");
    if (fp_out) {
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                fprintf(fp_out, "%17.14lf ", p[i][j]);
            }
            fprintf(fp_out, "\n");
        }
        fclose(fp_out);
    }

    /* Free memory */
    // TODO

    /* Finalize Hypre */
    HYPRE_Finalize();

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}
