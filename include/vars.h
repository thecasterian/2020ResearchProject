#ifndef VARS_H
#define VARS_H

typedef struct _part_mesh PartMesh;
typedef struct _initializer Initializer;

typedef struct _flow_vars {
    double *u1, *u1_next, *u1_hat;
    double *u2, *u2_next, *u2_hat;
    double *u3, *u3_next, *u3_hat;

    double *F1, *F1_next, *F1_hat;
    double *F2, *F2_next, *F2_hat;
    double *F3, *F3_next, *F3_hat;

    double *p, *p_next, *delta_p;

    double *H1, *H1_prev;
    double *H2, *H2_prev;
    double *H3, *H3_prev;

    int x_exchg_size, y_exchg_size, z_exchg_size;
    double *x_exchg, *y_exchg, *z_exchg;
} FlowVars;

FlowVars *FlowVars_Create(PartMesh *mesh);

void FlowVars_Initialize(FlowVars *vars, PartMesh *mesh, Initializer *init);

void FlowVars_UpdateOuter(FlowVars *vars, PartMesh *mesh);
void FlowVars_AdjExchg(FlowVars *vars, PartMesh *mesh);
void FlowVars_InterpF(FlowVars *vars, PartMesh *mesh);

void FlowVars_CalcH(FlowVars *vars, PartMesh *mesh);

void FlowVars_Destroy(FlowVars *vars);

#endif
