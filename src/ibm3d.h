#ifndef IBM3D_H
#define IBM3D_H

#include <stdbool.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include "geo3d.h"

typedef void *arr3d;
typedef void *arr4d;
typedef double (*IBMSolverInitFunc)(double, double, double);

typedef enum _linear_solver_type {
    SOLVER_AMG,
    SOLVER_PCG,
    SOLVER_BiCGSTAB,
    SOLVER_GMRES,
} IBMSolverLinearSolverType;

typedef enum _precond_type {
    PRECOND_NONE,
    PRECOND_AMG,
} IBMSolverPrecondType;

typedef enum _direction {
    DIR_NORTH = 1 << 0,
    DIR_EAST  = 1 << 1,
    DIR_SOUTH = 1 << 2,
    DIR_WEST  = 1 << 3,
    DIR_DOWN  = 1 << 4,
    DIR_UP    = 1 << 5
} IBMSolverDirection;

typedef enum _bc_type {
    BC_VELOCITY_INLET,          /* Normal velocity specified. */
    BC_PRESSURE_OUTLET,         /* Pressure specified. */
    BC_STATIONARY_WALL,         /* Stationary wall. */
    BC_FREE_SLIP_WALL,          /* Free-slip wall. */
    BC_ALL_PERIODIC,            /* All periodic. */
    BC_VELOCITY_PERIODIC,       /* Velocity periodic, pressure specified. */
} IBMSolverBCType;

typedef enum _bc_val_type {
    BC_CONST,                   /* Set boundary value by a constant. */
    BC_FUNC,                    /* Set boundary value by a function. */
} IBMSolverBCValType;

/* (t, x, y, z) => boundary value */
typedef double (*IBMSolverBCValFunc)(double, double, double, double);

typedef struct _bc {
    IBMSolverBCType type;
    IBMSolverBCValType val_type;
    union {
        double const_value;
        IBMSolverBCValFunc func;
    };
} IBMSolverBC;

enum {
    FLAG_FLUID = 1,
    FLAG_GHOST,
    FLAG_SOLID
};

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
    /* Min and max coordinates. */
    double xmin, xmax, ymin, ymax, zmin, zmax;

    /* Derivative coefficients. */
    double *kx_W, *kx_E, *ky_S, *ky_N, *kz_D, *kz_U;

    /* Boundary conditions of 6 outer boundaries: north, east, south, west, up,
       and down. */
    IBMSolverBC bc[6];

    /* Obstacle. */
    Polyhedron *poly;
    /* Flag of each cell (1: fluid cell, 2: ghost cell, 0: solid cell) */
    arr3d flag;
    /* Level set function. */
    arr3d lvset;

    /* Velocities and pressure. */
    arr3d u1, u1_next, u1_star, u1_tilde;
    arr3d u2, u2_next, u2_star, u2_tilde;
    arr3d u3, u3_next, u3_star, u3_tilde;

    arr3d U1, U1_next, U1_star;
    arr3d U2, U2_next, U2_star;
    arr3d U3, U3_next, U3_star;

    arr3d p, p_next, p_prime;

    /* Fluxes. */
    arr3d N1, N1_prev, N2, N2_prev, N3, N3_prev;

    /* Secondary halos. */
    arr3d vel_2nd_halo_lower;
    arr3d vel_2nd_halo_upper;

    /* HYPRE matrices, vectors, solvers, and arrays. */
    HYPRE_IJMatrix     A_u1, A_u2, A_u3, A_p;
    HYPRE_ParCSRMatrix parcsr_A_u1, parcsr_A_u2, parcsr_A_u3, parcsr_A_p;
    HYPRE_IJVector     b, x;
    HYPRE_ParVector    par_b, par_x;

    IBMSolverLinearSolverType linear_solver_type;
    IBMSolverPrecondType precond_type;

    HYPRE_Solver linear_solver, precond;
    HYPRE_Solver linear_solver_p, precond_p;
    double tol;

    arr3d p_coeffsum;

    int *vector_rows;
    double *vector_values, *vector_zeros, *vector_res;

    /* Iteration info. */
    int iter;
    double time;

    /* Autosave. */
    const char *autosave_filename;
    int autosave_period;
} IBMSolver;

#include "ibm3d_setup.h"
#include "ibm3d_fracstep.h"
#include "ibm3d_export.h"

#endif
