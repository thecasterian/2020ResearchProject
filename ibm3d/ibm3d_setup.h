#ifndef IBM3D_SETUP_H
#define IBM3D_SETUP_H

#include "ibm3d.h"

IBMSolver *IBMSolver_new(const int, const int);
void IBMSolver_destroy(IBMSolver *);

void IBMSolver_set_grid(
    IBMSolver *,
    const int, const int, const int,
    const double *restrict,
    const double *restrict,
    const double *restrict
);
void IBMSolver_set_params(IBMSolver *, const double, const double);

void IBMSolver_set_bc(IBMSolver *, IBMSolverDirection, IBMSolverBCType, double);
void IBMSolver_set_obstacle(IBMSolver *, Polyhedron *);
void IBMSolver_set_linear_solver(IBMSolver *, IBMSolverLinearSolverType, IBMSolverPrecondType, const double);

void IBMSolver_assemble(IBMSolver *);

void IBMSolver_init_flow_const(IBMSolver *);
void IBMSolver_init_flow_file(IBMSolver *, const char *, const char *, const char *, const char *);

void IBMSolver_export_results(IBMSolver *, const char *, const char *, const char *, const char *);

#endif
