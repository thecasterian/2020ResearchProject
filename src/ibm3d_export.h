#ifndef IBM3D_EXPORT_H
#define IBM3D_EXPORT_H

#include "ibm3d.h"

void IBMSolver_export_results(IBMSolver *, const char *, const char *, const char *, const char *);
void IBMSolver_export_netcdf3(IBMSolver *, const char *);

void IBMSolver_export_lvset(IBMSolver *, const char *);
void IBMSolver_export_flag(IBMSolver *, const char *);

#endif
