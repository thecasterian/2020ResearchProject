#ifndef IBM3D_INIT_H
#define IBM3D_INIT_H

#include "ibm3d.h"

void IBMSolver_init_flow_const(IBMSolver *, const double, const double, const double, const double);
void IBMSolver_init_flow_file(IBMSolver *, const char *);
void IBMSolver_init_flow_func(IBMSolver *, IBMSolverInitFunc, IBMSolverInitFunc, IBMSolverInitFunc, IBMSolverInitFunc);

#endif