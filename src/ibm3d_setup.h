#ifndef IBM3D_SETUP_H
#define IBM3D_SETUP_H

#include "ibm3d.h"

IBMSolver *IBMSolver_new(const int, const int, const int, const int, const int);
void IBMSolver_destroy(IBMSolver *);

void IBMSolver_set_grid(
    IBMSolver *,
    const int, const int, const int,
    const double *restrict,
    const double *restrict,
    const double *restrict
);
void IBMSolver_set_params(IBMSolver *, const double, const double);

void IBMSolver_set_ext_force(IBMSolver *, IBMSolverValType, ...);
void IBMSolver_set_turb_model(IBMSolver *, IBMSolverTurbModelType, ...);
void IBMSolver_set_bc(IBMSolver *, IBMSolverDirection, IBMSolverBCType, ...);
void IBMSolver_set_obstacle(IBMSolver *, Polyhedron *);
void IBMSolver_set_linear_solver(IBMSolver *, IBMSolverLinearSolverType, IBMSolverPrecondType, const double);
void IBMSolver_set_autosave(IBMSolver *, const char *, int);

void IBMSolver_assemble(IBMSolver *);

void IBMSolver_ghost_interp(
    IBMSolver *,
    const int, const int, const int,
    int [restrict][3], double [restrict]
);

#endif
