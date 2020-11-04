#ifndef IBM3D_SETUP_H
#define IBM3D_SETUP_H

#include "ibm3d.h"

IBMSolver *IBMSolver_new(const int, const int);
void IBMSolver_destroy(IBMSolver *);

void IBMSolver_set_grid_params(
    IBMSolver *,
    const int, const int, const int,
    const double *restrict,
    const double *restrict,
    const double *restrict,
    const double, const double
);
void IBMSolver_set_obstacle(IBMSolver *, Polyhedron *);

void IBMSolver_init_flow_const(IBMSolver *);
void IBMSolver_init_flow_file(IBMSolver *, FILE *, FILE *, FILE *, FILE *);

void IBMSolver_export_results(IBMSolver *, FILE *, FILE *, FILE *, FILE *);

#endif
