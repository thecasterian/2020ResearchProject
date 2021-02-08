#ifndef SOLVER_H
#define SOLVER_H

#include <stdbool.h>

typedef struct _part_mesh PartMesh;
typedef struct _params Params;
typedef struct _flow_vars FlowVars;

typedef struct _initializer Initializer;
typedef struct _netcdf_writer NetCDFWriter;

typedef struct _solver {
    PartMesh *mesh;
    Params *params;
    FlowVars *vars;

    double time;
    int iter;
} Solver;

Solver *Solver_Create(PartMesh *mesh, Params *params);

void Solver_Assemble(Solver *solver);
void Solver_Initialize(Solver *solver, Initializer *init);

void Solver_Iterate(Solver *solver, int ntimesteps, bool verbose);

void Solver_ExportNetCDF(Solver *solver, NetCDFWriter *writer);

void Solver_Destroy(Solver *solver);

#endif