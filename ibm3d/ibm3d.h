#ifndef IBM3D_H
#define IBM3D_H

#include <stdbool.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include "geo3d.h"

typedef double (*double3d)[];

typedef enum _linear_solver_type {
    SOLVER_AMG,
    SOLVER_PCG,
    SOLVER_BiCGSTAB,
    SOLVER_GMRES
} IBMSolverLinearSolverType;

typedef enum _precond_type {
    PRECOND_NONE = 10000,
    PRECOND_AMG
} IBMSolverPrecondType;

typedef struct _ibm_solver {
    /* Rank of current process. */
    int rank;
    /* Number of all processes. */
    int num_process;

    /* Grid dimensions. */
    int Nx_global, Ny, Nz;

    /* Min x-index of current process. */
    int ilower;
    /* Max x-index of current process. */
    int iupper;
    /* Number of x-index of current process. */
    int Nx;

    /* Reynolds number. */
    double Re;
    /* Delta t. */
    double dt;

    /* Cell widths. (local) */
    double *dx, *dy, *dz;
    /* Cell centeroid coordinates. (local) */
    double *xc, *yc, *zc;
    /* Global cell width and centroid coordinate. */
    double *dx_global, *xc_global;

    /* Flag of each cell (1: fluid cell, 2: ghost cell, 0: solid cell) */
    int (*flag)[];

    /* Velocities and pressure. */
    double3d u1, u1_next, u1_star, u1_tilde;
    double3d u2, u2_next, u2_star, u2_tilde;
    double3d u3, u3_next, u3_star, u3_tilde;

    double3d U1, U1_next, U1_star;
    double3d U2, U2_next, U2_star;
    double3d U3, U3_next, U3_star;

    double3d p, p_next, p_prime;

    /* Fluxes. */
    double3d N1, N1_prev, N2, N2_prev, N3, N3_prev;

    /* Derivative coefficients. */
    double *kx_W, *kx_E, *ky_S, *ky_N, *kz_D, *kz_U;

    /* HYPRE matrices, vectors, solvers, and arrays. */
    HYPRE_IJMatrix     A_u1, A_u2, A_u3, A_p;
    HYPRE_ParCSRMatrix parcsr_A_u1, parcsr_A_u2, parcsr_A_u3, parcsr_A_p;
    HYPRE_IJVector     b, x;
    HYPRE_ParVector    par_b, par_x;

    IBMSolverLinearSolverType linear_solver_type;
    IBMSolverPrecondType precond_type;

    HYPRE_Solver linear_solver, precond;
    HYPRE_Solver linear_solver_p, precond_p;

    int *vector_rows;
    double *vector_values, *vector_zeros, *vector_res;
} IBMSolver;

#include "ibm3d_setup.h"
#include "ibm3d_fracstep.h"

#endif
