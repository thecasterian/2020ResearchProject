#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include "geo3d.h"

#define SWAP(a, b) ({ __auto_type tmp = a; a = b; b = tmp;})

const int adj[6][3] = {
    {0, 1, 0}, {1, 0, 0}, {0, -1, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, 1}
};

int bs_upper_bound(const int len, double arr[const static len], const double val) {
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
    char stl_file[100];

    int Nx, Ny, Nz;
    double Re, dt;
    int numtstep;

    int init_using_file;
    char init_file_u1[100], init_file_u2[100], init_file_p[100];
    char output_file_u1[100], output_file_u2[100], output_file_p[100];

    FILE *fp_in = fopen("ibm3d.in", "r");
    if (!fp_in) {
        printf("cannot open file!\n");
        MPI_Finalize();
        return 0;
    }

    /* Read STL file name */
    fscanf(fp_in, "%*s %s", stl_file);

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

    fprintf(stderr, "Input mesh size: %d x %d x %d\n", Nx, Ny, Nz);

    /* Read Re, dt, numtstep */
    fscanf(fp_in, "%*s %lf", &Re);
    fscanf(fp_in, "%*s %lf %*s %d", &dt, &numtstep);

    /* File for initialization */
    fscanf(fp_in, "%*s %d", &init_using_file);
    fscanf(fp_in, "%*s %s", init_file_u1);
    fscanf(fp_in, "%*s %s", init_file_u2);
    fscanf(fp_in, "%*s %s", init_file_p);

    /* File for output */
    fscanf(fp_in, "%*s %s", output_file_u1);
    fscanf(fp_in, "%*s %s", output_file_u2);
    fscanf(fp_in, "%*s %s", output_file_p);

    fclose(fp_in);

    /*===== Define variables =================================================*/
    /* Polyhedron */
    Geo3dPolyhedron poly;

    /* Grid variables */
    double xc[Nx+2], dx[Nx+2];
    double yc[Ny+2], dy[Ny+2];
    double zc[Nz+2], dz[Nz+2];

    double (*const lvset)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    int (*const flag)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));
    int num_tc, num_gc;

    int (*cell_id)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(int [Ny+2][Nz+2]));
    int (*adj_cell_id)[Ny+2][Nz+2][6] = calloc(Nx+2, sizeof(int [Ny+2][Nz+2][6]));

    int (*gc_interp_cell_id)[8];
    double (*gc_interp_coeff)[8];

    /* Pressure and velocities */
    double (*      p       )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      p_next  )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const p_prime )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    double (*      u1      )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      u1_next )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const u1_star )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const u1_tilde)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    double (*      u2      )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      u2_next )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const u2_star )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const u2_tilde)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    double (*      u3      )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      u3_next )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const u3_star )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*const u3_tilde)[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    double (*      U1      )[Ny+2][Nz+2] = calloc(Nx+1, sizeof(double [Ny+2][Nz+2]));
    double (*      U1_next )[Ny+2][Nz+2] = calloc(Nx+1, sizeof(double [Ny+2][Nz+2]));
    double (*const U1_star )[Ny+2][Nz+2] = calloc(Nx+1, sizeof(double [Ny+2][Nz+2]));

    double (*      U2      )[Ny+1][Nz+2] = calloc(Nx+2, sizeof(double [Ny+1][Nz+2]));
    double (*      U2_next )[Ny+1][Nz+2] = calloc(Nx+2, sizeof(double [Ny+1][Nz+2]));
    double (*const U2_star )[Ny+1][Nz+2] = calloc(Nx+2, sizeof(double [Ny+1][Nz+2]));

    double (*      U3      )[Ny+2][Nz+1] = calloc(Nx+2, sizeof(double [Ny+2][Nz+1]));
    double (*      U3_next )[Ny+2][Nz+1] = calloc(Nx+2, sizeof(double [Ny+2][Nz+1]));
    double (*const U3_star )[Ny+2][Nz+1] = calloc(Nx+2, sizeof(double [Ny+2][Nz+1]));

    /* Fluxes */
    double (*      N1      )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      N1_prev )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      N2      )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      N2_prev )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      N3      )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));
    double (*      N3_prev )[Ny+2][Nz+2] = calloc(Nx+2, sizeof(double [Ny+2][Nz+2]));

    /* Others */
    double kx_W[Nx+2], kx_E[Nx+2];
    double ky_S[Ny+2], ky_N[Ny+2];
    double kz_D[Nz+2], kz_U[Nz+2];

    /* For HYPRE */
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

    /*===== Read polygon file ================================================*/
    FILE *fp_poly = fopen(stl_file, "rb");
    if (fp_poly) {
        Geo3dPolyhedron_init(&poly);
        Geo3dPolyhedron_read_stl(&poly, fp_poly);
    }

    /*===== Calculate grid variables =========================================*/
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

    /*===== Calculate level set function =====================================*/
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                Geo3dVector v = {xc[i], yc[j], zc[k]};
                lvset[i][j][k] = Geo3dPolyhedron_sgndist(&poly, v);
            }
        }
    }

    /*===== Calculate flags ==================================================*/
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
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
        }
    }

    /*===== Assign cell ids ==================================================*/
    num_tc = num_gc = 0;
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                cell_id[i][j][k] = 0;
            }
        }
    }

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                if (flag[i][j][k] == 2) {
                    num_gc++;
                    num_tc++;
                    cell_id[i][j][k] = num_tc;
                }
            }
        }
    }
    for (int i = 0; i <= Nx+1; i++) {
        for (int j = 0; j <= Ny+1; j++) {
            for (int k = 0; k <= Nz+1; k++) {
                if (
                       (i == 0    && j == 0    && k == 0   )
                    || (i == 0    && j == 0    && k == Nz+1)
                    || (i == 0    && j == Ny+1 && k == 0   )
                    || (i == 0    && j == Ny+1 && k == Nz+1)
                    || (i == Nx+1 && j == 0    && k == 0   )
                    || (i == Nx+1 && j == 0    && k == Nz+1)
                    || (i == Nx+1 && j == Ny+1 && k == 0   )
                    || (i == Nx+1 && j == Ny+1 && k == Nz+1)
                ) {
                    continue;
                }
                if (flag[i][j][k] == 1) {
                    num_tc++;
                    cell_id[i][j][k] = num_tc;
                }
            }
        }
    }

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            for (int k = 1; k <= Nz; k++) {
                if (flag[i][j][k] == 1) {
                    for (int l = 0; l < 6; l++) {
                        int ni = i + adj[l][0], nj = j + adj[l][1], nk = k + adj[l][2];
                        adj_cell_id[i][j][k][l] = cell_id[ni][nj][nk];
                    }
                }
            }
        }
    }

    fprintf(stderr, "# total cells: %d\n", num_tc);
    fprintf(stderr, "# ghost cells: %d\n", num_gc);

    /*===== Calculate interpolation infos for ghost cells ====================*/
    // TODO
}
