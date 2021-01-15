#ifndef IBM3D_EXPORT_H
#define IBM3D_EXPORT_H

#include "ibm3d.h"

void IBMSolver_export_result(IBMSolver *, const char *);

void IBMSolver_export_lvset_flag(IBMSolver *, const char *);
void IBMSolver_export_intermediate(IBMSolver *, const char *);

#endif
