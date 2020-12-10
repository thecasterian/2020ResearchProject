#ifndef IBM3D_H
#define IBM3D_H

#include <stdbool.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include "geo3d.h"

/* (x, y, z) => init value. */
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
    BC_VELOCITY_COMPONENT,      /* Velocity specified. */
    BC_PRESSURE,                /* Pressure specified. */
    BC_STATIONARY_WALL,         /* Stationary wall. */
    BC_FREE_SLIP_WALL,          /* Free-slip wall. */
    BC_SYMMETRIC,               /* Symmetric. */
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
        double const_u1, const_u2, const_u3, const_p;
        IBMSolverBCValFunc func_u1, func_u2, func_u3, func_p;
    };
} IBMSolverBC;

enum {
    FLAG_FLUID = 1,
    FLAG_GHOST,
    FLAG_SOLID
};

typedef struct _ibm_solver {
    /* Number of all process. */
    int num_process;
    /* Number of processes in each direction. */
    int Px, Py, Pz;
    /* Rank of current process. */
    int rank;
    /* Rank of current process in each direction. */
    int ri, rj, rk;

    /* Global grid dimensions. */
    int Nx_global, Ny_global, Nz_global;
    /* Local grid dimensions of current process. */
    int Nx, Ny, Nz;

    /* Min indices of current process. */
    int ilower, jlower, klower;
    /* Max indices of current process. */
    int iupper, jupper, kupper;

    /* Local grid dimensions including outer cells. */
    int Nx_out, Ny_out, Nz_out;
    /* Min and max indices including outer cells. */
    int ilower_out, jlower_out, klower_out;
    int iupper_out, jupper_out, kupper_out;

    /* First and last cell index. */
    int idx_first, idx_last;
    /* Cell indices. */
    int *cell_idx, *cell_idx_periodic;

    /* Reynolds number. */
    double Re;
    /* Delta t. */
    double dt;

    /* Iteration info. */
    int iter;
    double time;

    /* Autosave. */
    const char *autosave_filename;
    int autosave_period;

    /* Cell widths. (local) */
    double *dx, *dy, *dz;
    /* Cell centeroid coordinates. (local) */
    double *xc, *yc, *zc;
    /* Global cell widths and centroid coordinates. */
    double *dx_global, *dy_global, *dz_global;
    double *xc_global, *yc_global, *zc_global;
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
    int *flag;
    /* Level set function. */
    double *lvset;

    /* Velocities and pressure. */
    double *u1, *u1_next, *u1_star, *u1_tilde;
    double *u2, *u2_next, *u2_star, *u2_tilde;
    double *u3, *u3_next, *u3_star, *u3_tilde;

    double *U1, *U1_next, *U1_star;
    double *U2, *U2_next, *U2_star;
    double *U3, *U3_next, *U3_star;

    double *p, *p_next, *p_prime;

    /* Fluxes. */
    double *N1, *N1_prev, *N2, *N2_prev, *N3, *N3_prev;

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

    double *p_coeffsum;

    int *vector_rows;
    double *vector_values, *vector_zeros, *vector_res;

    /* Temporary array for exchange. */
    double *x_exchg, *y_exchg, *z_exchg;
} IBMSolver;

#include "ibm3d_setup.h"
#include "ibm3d_init.h"
#include "ibm3d_fracstep.h"
#include "ibm3d_export.h"

#endif
